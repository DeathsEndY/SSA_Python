import numpy as np
import datetime
from scipy.optimize import minimize_scalar
from scipy.signal import argrelextrema

def find_coarse_encounters(sat1, sat2, ts, start_time, end_time, step_sec, dist_threshold_km):
    """
    向量化粗搜函数，返回所有满足安全阈值的卫星交会时间列表
    """
    total_seconds = (end_time - start_time).total_seconds()
    offsets = np.arange(0, total_seconds, step_sec)
    dt_list = [start_time + datetime.timedelta(seconds=float(s)) for s in offsets]
    t_array = ts.from_datetimes(dt_list)

    # 批量计算距离
    r1_array = sat1.at(t_array).position.km
    r2_array = sat2.at(t_array).position.km
    distances = np.linalg.norm(r1_array - r2_array, axis=0)

    # 寻找谷底
    local_min_indices = argrelextrema(distances, np.less)[0]
    safe_coarse_threshold = dist_threshold_km + (20.0 * step_sec)

    coarse_tcas = []
    for idx in local_min_indices:
        if distances[idx] <= safe_coarse_threshold:
            coarse_tcas.append(dt_list[idx])
            
    return coarse_tcas


def refine_single_encounter(coarse_tca, sat1, sat2, ts, sat1_tle, sat2_tle, 
                             step_sec, dist_threshold_km, rtn_errors_km, hbr, reltol, hbr_type):
    """
    对单一粗搜时间点进行精搜：寻找精确 TCA 和 最大碰撞概率 Pc
    """
    from PcCalculate import tle_to_pc_at_time
    # 闭包 1：距离寻优目标函数
    def distance_at_offset(offset):
        t_eval = coarse_tca + datetime.timedelta(seconds=offset)
        r1 = sat1.at(ts.from_datetime(t_eval)).position.km
        r2 = sat2.at(ts.from_datetime(t_eval)).position.km
        return np.linalg.norm(r1 - r2)

    # 寻找精确 TCA
    res_dist = minimize_scalar(
        distance_at_offset, 
        bounds=(-step_sec, step_sec), 
        method='bounded',
        options={'xatol': 1e-4} 
    )
    exact_tca = coarse_tca + datetime.timedelta(seconds=res_dist.x)
    exact_min_dist = res_dist.fun
    
    # 如果精确距离没进阈值，直接返回 None
    if exact_min_dist > dist_threshold_km:
        return None

    # 闭包 2：概率寻优目标函数
    pc_search_window = 1.5 
    def negative_pc_at_offset(offset):
        t_eval = exact_tca + datetime.timedelta(seconds=offset)
        try:
            pc = tle_to_pc_at_time(
                sat1_tle, sat2_tle, t_eval.replace(tzinfo=None), 
                rtn_errors_km=rtn_errors_km, hbr=hbr, reltol=reltol, hbr_type=hbr_type
            )
            return -pc
        except Exception:
            return 0.0

    # 寻找最大 Pc
    res_pc = minimize_scalar(
        negative_pc_at_offset,
        bounds=(-pc_search_window, pc_search_window),
        method='bounded',
        options={'xatol': 1e-3}
    )

    max_pc_time = exact_tca + datetime.timedelta(seconds=res_pc.x)
    
    return {
        "tca_time": exact_tca.replace(tzinfo=None),
        "min_dist_km": exact_min_dist,
        "max_pc_time": max_pc_time.replace(tzinfo=None),
        "dist_at_max_pc": distance_at_offset(res_dist.x + res_pc.x),
        "max_pc": -res_pc.fun
    }