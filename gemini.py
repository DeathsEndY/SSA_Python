import numpy as np
import datetime
from datetime import timezone
from scipy.optimize import minimize_scalar
from skyfield.api import load, EarthSatellite

# ... 这里保留你原有的 cov_rem_eig_val_clip, pc2d_foster, generate_covariance_matrix, tle_to_pc_at_time 等函数 ...

def find_tca_and_max_pc(sat1_tle, sat2_tle, start_time, end_time, step_sec=10.0, 
                        rtn_errors_km=[1.0, 5.0, 1.0], hbr=0.15, reltol=1e-8, hbr_type="square"):
    """
    在给定的时间窗口内，以极低的时间复杂度寻找两颗卫星的最大碰撞概率及其发生时刻。
    
    参数:
    sat1_tle, sat2_tle: TLE 字典数据
    start_time: 开始时刻 (datetime 对象)
    end_time: 结束时刻 (datetime 对象)
    step_sec: 粗搜步长(秒)。建议10-30秒，足以捕捉低轨卫星交会。
    """
    ts = load.timescale()
    sat1 = EarthSatellite(sat1_tle["tle1"], sat1_tle["tle2"], sat1_tle["name"], ts)
    sat2 = EarthSatellite(sat2_tle["tle1"], sat2_tle["tle2"], sat2_tle["name"], ts)

    # 确保时间带有 UTC 时区，以便 Skyfield 批量处理
    if start_time.tzinfo is None:
        start_time = start_time.replace(tzinfo=timezone.utc)
    if end_time.tzinfo is None:
        end_time = end_time.replace(tzinfo=timezone.utc)

    total_seconds = (end_time - start_time).total_seconds()
    if total_seconds <= 0:
        raise ValueError("结束时间必须晚于开始时间")

    # ==========================================
    # 步骤 1: 向量化粗搜 (寻找相对距离的粗略极小值)
    # ==========================================
    # 生成时间数组
    offsets = np.arange(0, total_seconds, step_sec)
    dt_list = [start_time + datetime.timedelta(seconds=float(s)) for s in offsets]
    t_array = ts.from_datetimes(dt_list)

    # 批量计算位置并求距离 (极其高效)
    r1_array = sat1.at(t_array).position.km
    r2_array = sat2.at(t_array).position.km
    distances = np.linalg.norm(r1_array - r2_array, axis=0)

    # 找到距离最近的索引
    min_idx = np.argmin(distances)
    coarse_tca = dt_list[min_idx]
    coarse_min_dist = distances[min_idx]

    # 如果粗搜发现最近距离非常远 (比如 > 500km)，基本不可能发生碰撞，可提前阻断以节省算力
    # 但为了严谨，这里我们继续进行

    # ==========================================
    # 步骤 2: 局部精细寻优 (寻找精确的 TCA 时刻)
    # ==========================================
    def distance_at_offset(offset):
        """目标函数：输入相对于 coarse_tca 的时间偏移(秒)，返回两星距离"""
        t_eval = coarse_tca + datetime.timedelta(seconds=offset)
        t_sf = ts.from_datetime(t_eval)
        r1 = sat1.at(t_sf).position.km
        r2 = sat2.at(t_sf).position.km
        return np.linalg.norm(r1 - r2)

    # 在粗搜点前后各扩展一个步长的范围内，寻找精确极小值点
    res = minimize_scalar(
        distance_at_offset, 
        bounds=(-step_sec, step_sec), 
        method='bounded',
        options={'xatol': 1e-4} # 精度达到 0.1 毫秒即可
    )

    exact_tca = coarse_tca + datetime.timedelta(seconds=res.x)
    exact_min_dist = res.fun
    
    # 将时区感知对象转回与输入一致的 naive datetime，方便后续调用原有函数
    exact_tca_naive = exact_tca.replace(tzinfo=None)

    # ==========================================
    # 步骤 3: 仅在 TCA 时刻计算一次最大碰撞概率
    # ==========================================
    max_pc = tle_to_pc_at_time(
        sat1_tle, sat2_tle, exact_tca_naive, 
        rtn_errors_km=rtn_errors_km, hbr=hbr, reltol=reltol, hbr_type=hbr_type
    )

    return exact_tca_naive, max_pc, exact_min_dist


if __name__ == "__main__":
    # 目标1：STARLINK-3809 - 52851
    sat1_tle = {
        "name": "STARLINK-3809",
        "tle1": "1 52851U 22062X   26089.96624807  .00000045  00000-0  20931-4 0  9998",
        "tle2": "2 52851  53.2169 288.6068 0001274  89.8700 270.2438 15.08839245209139"
    }

    # 目标2：OBJECT T - 43775
    sat2_tle = {
        "name": "OBJECT T - 43775",
        "tle1": "1 43775U 18099T   26090.08321991  .00005708  00000-0  35881-3 0  9991",
        "tle2": "2 43775  97.4298 136.7548 0004620 223.1343 136.9525 15.09611681400532"
    }

    # 定义我们要搜索的时间窗口 (例如：2026年4月1日 13:00 到 14:00 这一个小时内)
    start_time = datetime.datetime(2026, 4, 1, 13, 0, 0)
    end_time = datetime.datetime(2026, 4, 1, 14, 0, 0)

    print(f"开始搜索分析窗口: {start_time} 到 {end_time} ...")
    
    # 调用优化算法寻找最大概率
    best_time, max_pc, min_distance = find_tca_and_max_pc(
        sat1_tle, sat2_tle, start_time, end_time, 
        step_sec=10.0, 
        rtn_errors_km=EMPIRICAL_ERRORS, 
        hbr=HBR, 
        reltol=RELTOL, 
        hbr_type=HBR_TYPE
    )

    print("="*50)
    print("分析结果完成！")
    print(f"▶ 最大碰撞概率发生时刻 (TCA) : {best_time}")
    print(f"▶ 该时刻的最短相对距离      : {min_distance:.3f} km")
    print(f"▶ 最大碰撞概率 (Max Pc)      : {max_pc:.12e}")
    print("="*50)