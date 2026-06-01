"""
safety.py
重构为更 Pythonic 的实现：dataclass，snake_case，清晰接口。
保留原有算法逻辑，但使用更明确的命名与类型提示。

交会分析由safety.py，近点与远点计算由closeApproach.py，卫星传播由satellite.py负责。

0529 更新：增加了时间窗筛选的占位函数，调整了输出格式，并添加了tle处理的错误输出和跳过机制。
0531 更新：原本处理卫星对部分数据冗余、序列化开销，现在精准传参，使用生成器和迭代器；增加概率计算和进度显示; 增加TCA附近最大概率寻优
0601 更新：增加Top10风险事件表格输出（终端+图片）; 每次只保存Top10的数据，不保存所有风险事件
"""
from __future__ import annotations

import sys
import time
import datetime
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')  # 防止在无 GUI 服务器上运行报错
import matplotlib.pyplot as plt
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Callable, List, Optional, Union
from scipy.optimize import minimize_scalar
from astropy.time import Time, TimeDelta

from closeApproach import ApproachInfo, CElement, find_arg_time, find_dec_time, GM_EARTH
from satellite import CSatellite
from PcCalculate import tle_to_pc_at_time

EMPIRICAL_ERRORS = [1.0, 5.0, 1.0]
HBR = 0.15
RELTOL = 1e-10
HBR_TYPE = "square"

@dataclass
class RiskEvent:
    tar_name: str
    con_name: str
    tar_id: int
    con_id: int
    tca: datetime.datetime
    rel_vel: float
    min_dist: float
    max_pc: float

@dataclass(frozen=True)
class SafetyConfig:
    threshold: float = 15.0
    day_window: float = 5.0
    output_path: Path = Path("output/safety_output_part_tle0531_cal.txt")
    tle_file: Path = Path("data/part_tle0531_cal.txt")


@dataclass
class TLEInfo:
    line1: str
    line2: str
    line3: str


@dataclass
class MissInfo:
    time: Time
    min_distance: float


def load_config(*, threshold: Optional[float] = None, day_window: Optional[float] = None,
                output_path: Optional[Union[str, Path]] = None,
                tle_file: Optional[Union[str, Path]] = None) -> SafetyConfig:
    """返回默认的 `SafetyConfig` 实例（不再读取外部 cfg 文件）。
    可通过关键字参数覆盖默认值。
    """
    cfg = SafetyConfig()
    kwargs = {}
    if threshold is not None:
        kwargs["threshold"] = float(threshold)
    if day_window is not None:
        kwargs["day_window"] = float(day_window)
    if output_path is not None:
        kwargs["output_path"] = Path(output_path)
    if tle_file is not None:
        kwargs["tle_file"] = Path(tle_file)
    if kwargs:
        return SafetyConfig(**{**cfg.__dict__, **kwargs})
    return cfg


def read_tar_tle(fname: Optional[Union[str, Path]] = None, config: Optional[SafetyConfig] = None) -> List[TLEInfo]:
    """
    从简单的文件格式读取 TLE：每三行为一组（line1,line2,line3）。
    若未提供 `fname`，则使用 `config.tle_file` 或 `SafetyConfig` 的默认值。
    """
    if fname is None:
        cfg = config or SafetyConfig()
        path = cfg.tle_file
    else:
        path = Path(fname)

    path = Path(path)
    if not path.exists():
        return []
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]

    entries: list[TLEInfo] = []
    seen: set[tuple[str, ...]] = set()
    for i in range(0, len(lines), 3):
        if i + 2 >= len(lines):
            break
        line1 = lines[i]
        line2 = lines[i + 1]
        line3 = lines[i + 2]
        fields = line2.split()
        if len(fields) >= 8:
            tle_key = tuple(fields[3:8])
        else:
            tle_key = (line2,)
        if tle_key in seen:
            duplicate_msg = (
                f"跳过重复 TLE：line1={line1}，line2 中间字段 {fields[3:8] if len(fields) >= 8 else line2} 与已有条目一致。"
            )
            print(duplicate_msg)
            if config is not None:
                try:
                    output_path = config.output_path
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    with output_path.open("a", encoding="utf-8") as f:
                        f.write(duplicate_msg + "\n")
                except Exception:
                    pass
            continue
        seen.add(tle_key)
        entries.append(TLEInfo(line1=line1, line2=line2, line3=line3))

    print("成功读取 TLE 条目数量:", len(entries))
    time.sleep(1)  # 模拟读取延迟

    return entries


def geometrical_filter1(tar_sat: CSatellite, con_sat: CSatellite, config: SafetyConfig) -> bool:
    """
    几何筛选1（原来的几何筛选）：判断近远地点区间是否可能重叠。
    返回 True 表示通过筛选（即可能相交），False 表示被筛除。
    """
    tar_ele = tar_sat.initial_element()
    con_ele = con_sat.initial_element()
    return not (
        tar_ele.perigee_radius > con_ele.apogee_radius + config.threshold
        or con_ele.perigee_radius > tar_ele.apogee_radius + config.threshold
    )


def _orbit_normal_vector(inclination_deg: float, raan_deg: float) -> np.ndarray:
    return np.array([
        np.sin(np.deg2rad(inclination_deg)) * np.sin(np.deg2rad(raan_deg)),
        -np.sin(np.deg2rad(inclination_deg)) * np.cos(np.deg2rad(raan_deg)),
        np.cos(np.deg2rad(inclination_deg)),
    ])


def _orbit_plane_basis(orb: CElement) -> tuple[np.ndarray, np.ndarray]:
    raan = orb.raan_rad
    incl = np.deg2rad(orb.inclination_deg)
    argp = orb.argp_rad
    cos_raan = np.cos(raan)
    sin_raan = np.sin(raan)
    cos_i = np.cos(incl)
    sin_i = np.sin(incl)
    cos_argp = np.cos(argp)
    sin_argp = np.sin(argp)

    rot = np.array([
        [cos_raan * cos_argp - sin_raan * cos_i * sin_argp,
         -cos_raan * sin_argp - sin_raan * cos_i * cos_argp,
         sin_raan * sin_i],
        [sin_raan * cos_argp + cos_raan * cos_i * sin_argp,
         -sin_raan * sin_argp + cos_raan * cos_i * cos_argp,
         -cos_raan * sin_i],
        [sin_i * sin_argp,
         sin_i * cos_argp,
         cos_i],
    ])

    p = rot @ np.array([1.0, 0.0, 0.0])
    q = rot @ np.array([0.0, 1.0, 0.0])
    return p, q


def _true_anomaly_for_direction(orb: CElement, direction: np.ndarray) -> float:
    p, q = _orbit_plane_basis(orb)
    d_unit = direction / np.linalg.norm(direction)
    x = np.dot(p, d_unit)
    y = np.dot(q, d_unit)
    return np.arctan2(y, x)


def _radius_at_true_anomaly(orb: CElement, nu: float) -> float:
    a = orb.semi_major_axis
    e = orb.ecc
    return a * (1 - e * e) / (1 + e * np.cos(nu))


def _orbit_position_at_true_anomaly(orb: CElement, nu: float) -> np.ndarray:
    p, q = _orbit_plane_basis(orb)
    r = _radius_at_true_anomaly(orb, nu)
    return r * (np.cos(nu) * p + np.sin(nu) * q)


def _distance_at_node_direction(
    tar_ele: CElement,
    con_ele: CElement,
    direction: np.ndarray,
) -> float:
    tar_nu = _true_anomaly_for_direction(tar_ele, direction)
    con_nu = _true_anomaly_for_direction(con_ele, direction)
    tar_pos = _orbit_position_at_true_anomaly(tar_ele, tar_nu)
    con_pos = _orbit_position_at_true_anomaly(con_ele, con_nu)
    return float(np.linalg.norm(tar_pos - con_pos))


def _time_from_true_anomaly(orb: CElement, target_nu: float) -> Time:
    e = orb.ecc
    nu0 = orb.true_anomaly_rad
    E0 = 2 * np.arctan(np.tan(nu0 / 2) * np.sqrt((1 - e) / (1 + e)))
    m0 = E0 - e * np.sin(E0)

    target_nu = (target_nu + np.pi) % (2 * np.pi) - np.pi
    E = 2 * np.arctan(np.tan(target_nu / 2) * np.sqrt((1 - e) / (1 + e)))
    m = E - e * np.sin(E)

    dM = (m - m0 + np.pi) % (2 * np.pi) - np.pi
    dt = dM / orb.mean_motion
    return orb.epoch + TimeDelta(dt, format="sec")


def _normalize_time_to_epoch(time_point: Time, base_epoch: Time, period: float) -> Time:
    offset = (base_epoch - time_point).sec
    n = int(np.round(offset / period))
    return time_point + TimeDelta(n * period, format="sec")


def _intersection_line_direction(tar_ele: CElement, con_ele: CElement) -> np.ndarray:
    n_tar = _orbit_normal_vector(tar_ele.inclination_deg, tar_ele.raan_deg)
    n_con = _orbit_normal_vector(con_ele.inclination_deg, con_ele.raan_deg)
    line_dir = np.cross(n_tar, n_con)
    norm = np.linalg.norm(line_dir)
    if norm < 1e-8:
        return np.zeros(3)
    return line_dir / norm


def _nearest_crossing_window(
    orb: CElement,
    line_dir: np.ndarray,
    threshold: float,
) -> tuple[Time, Time]:
    # 保留一个简易中心时间计算，但不再用于窗口宽度计算。
    nu = _true_anomaly_for_direction(orb, line_dir)
    t1 = _time_from_true_anomaly(orb, nu)
    t2 = _time_from_true_anomaly(orb, nu + np.pi)
    t1 = _normalize_time_to_epoch(t1, orb.epoch, orb.period)
    t2 = _normalize_time_to_epoch(t2, orb.epoch, orb.period)
    center = t1 if abs((t1 - orb.epoch).sec) <= abs((t2 - orb.epoch).sec) else t2
    # 返回基于单星速度的窗口（向后兼容），但主流程会使用基于相对速度的窗口
    speed = np.linalg.norm(orb.vel)
    half_window = threshold / max(speed, 1e-6)
    return (center - TimeDelta(half_window, format="sec"), center + TimeDelta(half_window, format="sec"))


def _window_candidate_periods(
    tar_ele: CElement,
    con_ele: CElement,
    config: SafetyConfig,
    num_periods: int,
) -> list[int]:
    """
    计算基于相位差的有效时间窗：
    返回目标卫星 (tar_sat) 可能发生危险交会的周期索引列表。
    加入动态时间漂移补偿，防止多天预测时的相位误差导致漏筛。
    """
    line_dir = _intersection_line_direction(tar_ele, con_ele)
    # 如果轨道近似平行或共面，保守起见返回所有周期
    if np.linalg.norm(line_dir) < 1e-8:
        return list(range(num_periods))

    line_dir_unit = line_dir / np.linalg.norm(line_dir)
    valid_indices = set()
    
    T_tar = tar_ele.period
    T_con = con_ele.period

    # 必须检查交线的两个穿透点 (方向与反方向)
    for direction in [line_dir_unit, -line_dir_unit]:
        nu_tar = _true_anomaly_for_direction(tar_ele, direction)
        nu_con = _true_anomaly_for_direction(con_ele, direction)
        
        t_tar_0 = _time_from_true_anomaly(tar_ele, nu_tar)
        t_con_0 = _time_from_true_anomaly(con_ele, nu_con)
        
        # 将基准交点时间归一化到 epoch 附近
        t_tar_0 = _normalize_time_to_epoch(t_tar_0, tar_ele.epoch, T_tar)
        t_con_0 = _normalize_time_to_epoch(t_con_0, con_ele.epoch, T_con)
        
        # Vis-viva 方程估算交点处的轨道速度
        r_tar = _radius_at_true_anomaly(tar_ele, nu_tar)
        r_con = _radius_at_true_anomaly(con_ele, nu_con)
        v_tar = np.sqrt(GM_EARTH * (2.0 / r_tar - 1.0 / tar_ele.semi_major_axis))
        v_con = np.sqrt(GM_EARTH * (2.0 / r_con - 1.0 / con_ele.semi_major_axis))
        
        # 相对速度最大保守估计
        rel_v = float(v_tar + v_con)
        
        # 计算基础安全时间窗 (稍微放宽基础 safety_factor，原为 150)
        safety_factor = 250
        base_half_window_sec = (config.threshold * safety_factor) / max(rel_v, 1e-6)
        
        # 遍历目标星的每一圈 n
        for n in range(num_periods):
            # 目标卫星第 n 圈到达交点的时间
            t_tar_n = t_tar_0 + TimeDelta(n * T_tar, format="sec")
            
            # 计算次星到达交点最接近 t_tar_n 的那一圈 (m)
            delta_t_sec = (t_tar_n - t_con_0).sec
            m = round(delta_t_sec / T_con)
            
            # 【核心修复】：引入随时间（圈数）增长的漂移补偿 (Drift Margin)
            # 假设 SGP4 与理想开普勒周期的累积误差为每圈约增加 3~6 秒（LEO 卫星典型值）
            # 根据主星和次星各自累积的圈数，动态放大安全时间窗
            drift_margin = abs(n) * 10.0 + abs(m) * 10.0 
            current_half_window = base_half_window_sec + drift_margin
            
            # 为了防止浮点误差和边界情况，检查最接近的 3 圈 (m-1, m, m+1)
            for dm in [-1, 0, 1]:
                closest_t_con = t_con_0 + TimeDelta((m + dm) * T_con, format="sec")
                diff_sec = abs((t_tar_n - closest_t_con).sec)
                
                # 使用动态放大的窗口进行判断
                if diff_sec <= current_half_window:
                    valid_indices.add(n)
                    break  # 当前 n 圈已被标记为危险，跳出内部 dm 循环
                    
    # 返回排序后的、可能发生交会的目标星圈数列表
    return sorted(list(valid_indices))


def geometrical_filter2(tar_sat: CSatellite, con_sat: CSatellite, config: SafetyConfig) -> bool:
    """
    几何筛选2：基于两个轨道平面的交线，计算交点处轨道间的最小距离。
    该方法先解析计算交线方向，再在交线上求两轨道对应点的距离。
    返回 True 表示通过筛选（可能相近），False 表示被筛除（轨道最近距离大于阈值）。
    """
    tar_ele = tar_sat.initial_element()
    con_ele = con_sat.initial_element()

    n_tar = _orbit_normal_vector(tar_ele.inclination_deg, tar_ele.raan_deg)
    n_con = _orbit_normal_vector(con_ele.inclination_deg, con_ele.raan_deg)
    line_dir = np.cross(n_tar, n_con)
    norm = np.linalg.norm(line_dir)
    if norm < 1e-8:
        # 两轨道面近似共面或平行，暂时不在此处筛除，交点搜索失效时保守处理。
        return True
    line_dir /= norm

    dist1 = _distance_at_node_direction(tar_ele, con_ele, line_dir)
    dist2 = _distance_at_node_direction(tar_ele, con_ele, -line_dir)
    min_distance = min(dist1, dist2)
    # 需要放宽阈值
    threshold = config.threshold * 1.5
    return min_distance <= threshold


def time_window_filter(tar_sat: CSatellite, con_sat: CSatellite, config: SafetyConfig) -> bool:
    """时间窗筛选：基于轨道面交线经过时间的重叠窗口。
    每个卫星在轨道面交线交点附近的时间段为 [t - T/2, t + T/2]，
    T 由距离阈值和卫星速度计算得出。
    如果两个窗口无交集，则返回 False；否则返回 True。
    """
    tar_ele = tar_sat.initial_element()
    con_ele = con_sat.initial_element()
    line_dir = _intersection_line_direction(tar_ele, con_ele)
    if np.linalg.norm(line_dir) < 1e-8:
        return True
    tar_start, tar_end = _nearest_crossing_window(tar_ele, line_dir, config.threshold)
    con_start, con_end = _nearest_crossing_window(con_ele, line_dir, config.threshold)
    return not (tar_end < con_start or con_end < tar_start)


def _write_close_approach(
    tar_sat: CSatellite,
    con_sat: CSatellite,
    miss: MissInfo,
    rel_vel: float,
    pc:float,
    output_path: Path,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as f:
        f.write(
            f"主目标{tar_sat.name} ID: {tar_sat.sat_id}, 次目标{con_sat.name} ID: {con_sat.sat_id} - "
            f"时间 {miss.time.utc.iso}, 最近距离 {miss.min_distance:.4f} km, 相对速度 {rel_vel:.3f} km/s, 碰撞概率 {pc:.4e}\n"
        )
    return 1

def _evaluate_approach_mode(
    tar_sat: CSatellite,
    con_sat: CSatellite,
    tar_info: TLEInfo,
    con_info: TLEInfo,
    period_indices: list[int],
    config: SafetyConfig,
    approach_fn: Callable[[CSatellite, CSatellite, int], 'ApproachInfo'],
) -> list[RiskEvent]:  # 修改返回值类型

    events = []
    for i in period_indices:
        approach_info = approach_fn(tar_sat, con_sat, i)
        
        miss = find_miss_distance(tar_sat, con_sat, approach_info.time)
        rel_vel = np.linalg.norm(
            tar_sat.propagate(miss.time).vel - con_sat.propagate(miss.time).vel
        )
        
        if miss.min_distance < config.threshold:
            sat1_tle_dict = {"name": tar_info.line1, "tle1": tar_info.line2, "tle2": tar_info.line3}
            sat2_tle_dict = {"name": con_info.line1, "tle1": con_info.line2, "tle2": con_info.line3}
            
            tca_dt = miss.time.to_datetime().replace(tzinfo=None)
            pc_search_window_sec = 1.5  
            
            def negative_pc_at_offset(offset: float) -> float:
                t_eval = tca_dt + datetime.timedelta(seconds=float(offset))
                try:
                    pc_val = tle_to_pc_at_time(
                        sat1_tle_dict, sat2_tle_dict, t_eval,
                        rtn_errors_km=EMPIRICAL_ERRORS,
                        hbr=HBR,
                        reltol=RELTOL,
                        hbr_type=HBR_TYPE
                    )
                    return -pc_val
                except Exception:
                    return 0.0

            try:
                res_pc = minimize_scalar(
                    negative_pc_at_offset,
                    bounds=(-pc_search_window_sec, pc_search_window_sec),
                    method='bounded',
                    options={'xatol': 1e-3}  
                )
                max_pc = -res_pc.fun
            except Exception:
                max_pc = -negative_pc_at_offset(0.0)

            # 写入日志
            _write_close_approach(
                tar_sat, con_sat, miss, rel_vel, max_pc, config.output_path
            )
            
            # 将该事件记录到列表中
            events.append(RiskEvent(
                tar_name=tar_sat.name,
                con_name=con_sat.name,
                tar_id=tar_sat.sat_id,
                con_id=con_sat.sat_id,
                tca=tca_dt,
                rel_vel=rel_vel,
                min_dist=miss.min_distance,
                max_pc=max_pc
            ))
            
    return events

def evaluate_pair(
        tar_sat: CSatellite, 
        con_sat: CSatellite,
        tar_info: TLEInfo, 
        con_info: TLEInfo,
        config: SafetyConfig
) -> tuple[int, int, int, int, list[RiskEvent]]:
    """
    多层筛选，若不通过返回 1，否则执行近点和远点计算。
    0601：增加了返回值中的风险事件列表，供后续统计和表格输出使用。
    """
    # 几何筛选1
    if not geometrical_filter1(tar_sat, con_sat, config):
        # (filtered_flag, geo1_filtered, geo2_filtered, time_filtered)
        return (1, 1, 0, 0, [])

    # 几何筛选2（基于轨道最近距离）
    if not geometrical_filter2(tar_sat, con_sat, config):
        return (1, 0, 1, 0, [])
    
    tar_ele = tar_sat.initial_element()
    con_ele = con_sat.initial_element()
    num_periods = max(1, int(config.day_window * 86400.0 / tar_ele.period))

    # 时间窗筛选
    valid_periods = _window_candidate_periods(tar_ele, con_ele, config, num_periods)
    # valid_periods = list(range(num_periods))   # 不进行时间窗筛选，验证算法时使用
    # print(f"valid_periods: {valid_periods}" )
    if len(valid_periods) == 0:
        return (1, 0, 0, 1, [])

    events = []
    events.extend(_evaluate_approach_mode(tar_sat, con_sat, tar_info, con_info, valid_periods, config, find_arg_time))
    events.extend(_evaluate_approach_mode(tar_sat, con_sat, tar_info, con_info, valid_periods, config, find_dec_time))

    return (0, 0, 0, 0, events)


def find_miss_distance(tar_sat: CSatellite, con_sat: CSatellite, time_point: Time) -> MissInfo:
    """
    使用牛顿迭代法求两个卫星在局部最短距离时刻并返回距离与时间
    """
    true_time = find_root_newton(tar_sat, con_sat, time_point, tol=0.1)
    tar_e = tar_sat.propagate(true_time)
    con_e = con_sat.propagate(true_time)
    rel_pos = con_e.pos - tar_e.pos
    return MissInfo(time=true_time, min_distance=float(np.linalg.norm(rel_pos)))


def find_root_newton(tar_sat: CSatellite, con_sat: CSatellite, initial_time: Time, tol: float) -> Time:
    h = 0.1
    current = initial_time
    delta = tol + 1.0
    max_iter = 20
    it = 0
    while abs(delta) > tol and it < max_iter:
        f = rtn_dot_product(tar_sat, con_sat, current)
        f1 = rtn_dot_product(tar_sat, con_sat, current + TimeDelta(h, format="sec"))
        df = (f1 - f) / h
        if abs(df) < 1e-12:
            print("Warning: derivative too small, stop Newton iteration.")
            break
        delta = f / df
        current = current - TimeDelta(delta, format="sec")
        it += 1

    if it == max_iter:
        print("\nWarning: Newton iteration reached max iterations; result may be imprecise.")
    return current


def rtn_dot_product(tar_sat: CSatellite, con_sat: CSatellite, tp: Time) -> float:
    tar_e = tar_sat.propagate(tp)
    con_e = con_sat.propagate(tp)
    rel_r = con_e.pos - tar_e.pos
    rel_v = con_e.vel - tar_e.vel
    return float(np.dot(rel_v, rel_r))


def relative_rtn(tar_rv, con_rv):
    """
    把相对向量转换到 RTN 坐标（保留原名以供可能的外部调用）。
    """
    rr = tar_rv[:3]
    vv = tar_rv[3:]
    nn = np.cross(rr, vv)
    tt = np.cross(nn, rr)
    mat = np.column_stack([rr / np.linalg.norm(rr), tt / np.linalg.norm(tt), nn / np.linalg.norm(nn)])
    return np.dot(mat.T, tar_rv[:3] - con_rv[:3])


# def _print_top10_tables(top_dist: list[RiskEvent], top_pc: list[RiskEvent]) -> None:
#     if not top_dist and not top_pc:
#         print("\n没有发现满足阈值的危险接近事件。")
#         return
        
#     print("\n\n" + "="*105)
#     print(" Top 10 危险事件 - 按【最近距离】排序 (距离从小到大)")
#     print("="*105)
#     print(f"{'主目标 (NORAD ID)':<22} | {'次目标 (NORAD ID)':<22} | {'TCA (UTC)':<20} | {'速度(km/s)':<10} | {'距离(km)':<10} | {'碰撞概率(Pc)':<12}")
#     print("-" * 105)
    
#     # # 按照距离升序排序并取前 10
#     # top_dist = sorted(events, key=lambda x: x.min_dist)[:10]
#     for e in top_dist:
#         tar_str = f"{e.tar_name[:13]} ({e.tar_id})"
#         con_str = f"{e.con_name[:13]} ({e.con_id})"
#         print(f"{tar_str:<22} | {con_str:<22} | {e.tca.strftime('%Y-%m-%d %H:%M:%S'):<20} | {e.rel_vel:<10.3f} | {e.min_dist:<10.3f} | {e.max_pc:<12.4e}")
              
#     print("\n" + "="*105)
#     print(" Top 10 危险事件 - 按【碰撞概率】排序 (概率从大到小)")
#     print("="*105)
#     print(f"{'主目标 (NORAD ID)':<22} | {'次目标 (NORAD ID)':<22} | {'TCA (UTC)':<20} | {'速度(km/s)':<10} | {'距离(km)':<10} | {'碰撞概率(Pc)':<12}")
#     print("-" * 105)
    
#     # # 按照碰撞概率降序排序并取前 10
#     # top_pc = sorted(events, key=lambda x: x.max_pc, reverse=True)[:10]
#     for e in top_pc:
#         tar_str = f"{e.tar_name[:13]} ({e.tar_id})"
#         con_str = f"{e.con_name[:13]} ({e.con_id})"
#         print(f"{tar_str:<22} | {con_str:<22} | {e.tca.strftime('%Y-%m-%d %H:%M:%S'):<20} | {e.rel_vel:<10.3f} | {e.min_dist:<10.3f} | {e.max_pc:<12.4e}")

def _print_top10_tables(top_dist: list[RiskEvent], top_pc: list[RiskEvent], output_dir: str = "output/gragh") -> None:
    
    # === 1. 终端文本输出 ===
    if not top_dist and not top_pc:
        print("\n没有发现满足阈值的危险接近事件。")
        return
        
    print("\n\n" + "="*105)
    print(" Top 10 危险事件 - 按【最近距离】排序 (距离从小到大)")
    print("="*105)
    print(f"{'主目标 (NORAD ID)':<22} | {'次目标 (NORAD ID)':<22} | {'TCA (UTC)':<20} | {'速度(km/s)':<10} | {'距离(km)':<10} | {'碰撞概率(Pc)':<12}")
    print("-" * 105)
    
    for e in top_dist:
        tar_str = f"{e.tar_name[:13]} ({e.tar_id})"
        con_str = f"{e.con_name[:13]} ({e.con_id})"
        print(f"{tar_str:<22} | {con_str:<22} | {e.tca.strftime('%Y-%m-%d %H:%M:%S'):<20} | {e.rel_vel:<10.3f} | {e.min_dist:<10.3f} | {e.max_pc:<12.4e}")
              
    print("\n" + "="*105)
    print(" Top 10 危险事件 - 按【碰撞概率】排序 (概率从大到小)")
    print("="*105)
    print(f"{'主目标 (NORAD ID)':<22} | {'次目标 (NORAD ID)':<22} | {'TCA (UTC)':<20} | {'速度(km/s)':<10} | {'距离(km)':<10} | {'碰撞概率(Pc)':<12}")
    print("-" * 105)
    
    for e in top_pc:
        tar_str = f"{e.tar_name[:13]} ({e.tar_id})"
        con_str = f"{e.con_name[:13]} ({e.con_id})"
        print(f"{tar_str:<22} | {con_str:<22} | {e.tca.strftime('%Y-%m-%d %H:%M:%S'):<20} | {e.rel_vel:<10.3f} | {e.min_dist:<10.3f} | {e.max_pc:<12.4e}")

    # === 2. 导出图片输出 ===
    os.makedirs(output_dir, exist_ok=True)
    
    def save_table_img(data_list, title, filename, is_pc_sort=False):
        if not data_list:
            return
            
        headers = [
            "Primary Satellite\n(NORAD ID)", 
            "Secondary Satellite\n(NORAD ID)", 
            "TCA (UTC)", 
            "Rel Vel\n(km/s)", 
            "Min Dist\n(km)", 
            "Prob of Collision\n(Pc)"
        ]
        
        cell_text = []
        for e in data_list:
            cell_text.append([
                f"{e.tar_name}\n({e.tar_id})",
                f"{e.con_name}\n({e.con_id})",
                e.tca.strftime("%Y-%m-%d\n%H:%M:%S"),
                f"{e.rel_vel:.3f}",
                f"{e.min_dist:.4f}",
                f"{e.max_pc:.4e}"
            ])
            
        fig, ax = plt.subplots(figsize=(12, 6), dpi=200)
        ax.axis('off')
        ax.axis('tight')
        
        table = ax.table(
            cellText=cell_text, 
            colLabels=headers, 
            cellLoc='center', 
            loc='center'
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.0, 2.2) # 略微拉高行距，显得不拥挤
        
        # 样式颜色
        header_bg = '#1F4E79'
        header_fg = '#FFFFFF'
        row_bg_even = '#F2F4F7'
        row_bg_odd = '#FFFFFF'
        # 第一排危险事件的高亮：距离表用警告色(浅橙)，概率表用警惕色(浅绿)
        highlight_bg = '#FCE4D6' if not is_pc_sort else '#E2EFDA' 
        
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_text_props(weight='bold', color=header_fg)
                cell.set_facecolor(header_bg)
            else:
                if row % 2 == 0:
                    cell.set_facecolor(row_bg_even)
                else:
                    cell.set_facecolor(row_bg_odd)
                
                # Top 1 极度高亮
                if row == 1:
                    cell.set_facecolor(highlight_bg)
                    cell.set_text_props(weight='bold')
                    
            cell.set_edgecolor('#D9D9D9')
            cell.set_linewidth(0.8)
            
        plt.title(title, fontsize=14, weight='bold', pad=20, color='#333333')
        
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, bbox_inches='tight', pad_inches=0.3)
        plt.close()
        print(f"成功保存报告图片到: {filepath}")

    # 调用绘图子函数
    save_table_img(top_dist, "Top 10 Conjunction Events - Sorted by Minimum Distance", "top10_min_dist.png", is_pc_sort=False)
    save_table_img(top_pc, "Top 10 Conjunction Events - Sorted by Collision Probability (Pc)", "top10_max_pc.png", is_pc_sort=True)

def process_satellite_pair(args):
    '''
    处理单个卫星对的入口函数：从TLE信息创建卫星对象，执行评估，并处理可能的异常。
    '''
    i, info_i, j, info_j, config = args
    # print(f"处理卫星对: [{i}] {info_i.line1} & [{j}] {info_j.line1}")

    try:
        tar = CSatellite(info_i.line1, info_i.line2, info_i.line3)
        con = CSatellite(info_j.line1, info_j.line2, info_j.line3)
        return evaluate_pair(tar, con, info_i, info_j, config)
    except Exception as e:
        err_msg = f"跳过卫星对 [{i} {info_i.line1} & {j} {info_j.line1}]，传播失败: {e}"
        print(err_msg)
        # 将错误也写入输出文件（若 config 有效）
        try:
            out_path = config.output_path if config and hasattr(config, "output_path") else SafetyConfig().output_path
            out_path = Path(out_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("a", encoding="utf-8") as f:
                f.write(err_msg + "\n")
        except Exception:
            # 若写文件也失败，则静默忽略以避免二次异常
            pass
        return (1, 0, 0, 0, [])


def process_all_satellite_pairs(config: SafetyConfig | None = None) -> None:
    '''
    批量处理所有卫星对：从 TLE 文件中读取数据，生成所有唯一对，调用process_satellite_pair并行评估每对的安全性。
    '''
    start = time.time()
    config = config or load_config()
    tles = read_tar_tle(config=config)
    n = len(tles)
    if n < 2:
        print("Error: 需要至少两个卫星的TLE数据。")
        sys.exit(1)

    total_pairs = n * (n - 1) // 2

    def generate_tasks():
        for i in range(n):
            for j in range(i + 1, n):
                yield (i, tles[i], j, tles[j], config)

    results = []
    # 两个独立列表维护 Top 10事件
    top10_min_dist: list[RiskEvent] = []
    top10_max_pc: list[RiskEvent] = []
    
    last_logged_milestone = -1  # 记录上一次写入文件的进度里程碑

    print("开始处理卫星对，进度如下：")
    with Pool(processes=8) as pool:
        iterator = pool.imap_unordered(process_satellite_pair, generate_tasks(), chunksize=1000)
        
        for idx, result in enumerate(iterator, start=1):
            results.append(result[:4])
            events = result[4]
            
            # 只维护 Top 10 变量，防内存溢出
            if events:
                # 维护最小距离 Top 10
                top10_min_dist.extend(events)
                top10_min_dist.sort(key=lambda x: x.min_dist)
                top10_min_dist = top10_min_dist[:10]
                
                # 维护最大概率 Top 10
                top10_max_pc.extend(events)
                top10_max_pc.sort(key=lambda x: x.max_pc, reverse=True)
                top10_max_pc = top10_max_pc[:10]
            
            # 1. 计算当前的精确百分比
            percent = (idx / total_pairs) * 100
            
            # 2. 在终端实时同行覆盖刷新
            print(f"\r进度: [{idx}/{total_pairs}] {percent:.2f}% 完成", end="", flush=True)
            
            # 3. 在输出文件中里程碑式记录 (这里设置为每 5% 记录一次)
            # 通过取整来判断是否到达新的 5% 节点
            milestone = int(percent // 5) * 5 
            if milestone % 5 == 0 and milestone != last_logged_milestone:
                last_logged_milestone = milestone
                progress_msg = f"--- 进度报告: 已完成 {milestone}% ({idx}/{total_pairs}) ---\n"
                
                try:
                    config.output_path.parent.mkdir(parents=True, exist_ok=True)
                    with config.output_path.open("a", encoding="utf-8") as f:
                        f.write(progress_msg)
                except Exception:
                    pass # 若写入失败静默处理，避免中断主计算流程 

    print() # 进度条结束后输出一个换行符，防止后续统计数据粘在同一行

    # --- 后续的统计和打印代码保持不变 ---
    total_filtered = sum(r[0] for r in results)
    num_geo1_filtered = sum(r[1] for r in results)
    num_geo2_filtered = sum(r[2] for r in results)
    num_time_filtered = sum(r[3] for r in results)
    end = time.time()
    
    _print_top10_tables(top10_min_dist, top10_max_pc)
    print(f"共处理 {n} 个目标，共处理 {total_pairs} 对")
    print(f"几何筛选1共有 {num_geo1_filtered} 对被筛除，占 {num_geo1_filtered / total_pairs * 100:.2f}%")
    print(f"几何筛选2共有 {num_geo2_filtered} 对被筛除，占 {num_geo2_filtered / total_pairs * 100:.2f}%")
    print(f"时间窗筛选共有 {num_time_filtered} 对被筛除，占 {num_time_filtered / total_pairs * 100:.2f}%")
    print(f"总共被筛除 {total_filtered} 对，占 {total_filtered / total_pairs * 100:.2f}%")
    print(f"耗时 {end - start:.2f}s, 平均 {(end - start) / total_pairs:.3f}s/次")

def single_satellite_safety(config: SafetyConfig | None = None) -> None:
    '''
    单星评估：针对一个目标卫星进行接近分析。
    '''
    config = config or load_config()
    start = time.time()
    tles = read_tar_tle(config=config)
    if len(tles) < 2:
        print("Error: 需要至少两个卫星的TLE数据。")
        return
    # 选择第一个卫星作为目标
    tar_info = tles[0]
    tar = CSatellite(tar_info.line1, tar_info.line2, tar_info.line3)

    geo1_filtered = 0
    geo2_filtered = 0
    time_filtered = 0
    total_filtered = 0
    pair_count = 0
    
    top10_min_dist: list[RiskEvent] = []
    top10_max_pc: list[RiskEvent] = []

    for tle in tles[1:]:
        pair_count += 1
        print(f"处理卫星: [{pair_count}] {tle.line1}")
        con_info = tle
        con = CSatellite(con_info.line1, con_info.line2, con_info.line3)
        
        f, g1, g2, t, events = evaluate_pair(tar, con, tar_info, con_info, config)
        total_filtered += f
        geo1_filtered += g1
        geo2_filtered += g2
        time_filtered += t
        
        if events:
            top10_min_dist.extend(events)
            top10_min_dist.sort(key=lambda x: x.min_dist)
            top10_min_dist = top10_min_dist[:10]
            
            top10_max_pc.extend(events)
            top10_max_pc.sort(key=lambda x: x.max_pc, reverse=True)
            top10_max_pc = top10_max_pc[:10]

    end = time.time()
    _print_top10_tables(top10_min_dist, top10_max_pc)
    print(f"共处理 {len(tles)} 个目标，共处理 {pair_count} 对")
    print(f"几何筛选1共有 {geo1_filtered} 对被筛除，占 {geo1_filtered / pair_count * 100:.2f}%")
    print(f"几何筛选2共有 {geo2_filtered} 对被筛除，占 {geo2_filtered / pair_count * 100:.2f}%")
    print(f"时间窗筛选共有 {time_filtered} 对被筛除，占 {time_filtered / pair_count * 100:.2f}%")
    print(f"总共被筛除 {total_filtered} 对，占 {total_filtered / pair_count * 100:.2f}%")
    print(f"耗时 {end - start:.2f}s, 平均 {(end - start) / pair_count:.3f}s/次")


if __name__ == "__main__":
    config = load_config()
    
    # 提示输入选择
    choice = input("请选择处理方式（1/2）\n1：处理所有卫星对\n2：处理单个目标卫星\n请输入你的选择：")
    
    # 根据输入执行相应的函数
    if choice == '1':
        print("正在运行：处理所有卫星对...")
        process_all_satellite_pairs(config)
    elif choice == '2':
        print("正在运行：处理单个目标卫星...")
        single_satellite_safety(config)
    else:
        print("输入无效，请重新运行程序并输入 1 或 2。")