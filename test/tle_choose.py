import math
from sgp4.api import Satrec, WGS84

# 地球物理常数
MU_EARTH = 398600.4418  # 地球标准引力参数 (km^3/s^2)
R_EARTH = 6371.0        # 地球平均半径 (km)

def is_tle_valid(line1: str, line2: str) -> bool:
    """
    根据物理合理性和 SGP4 传播状态筛选 TLE 数据。
    返回 True 表示有效，False 表示异常需剔除。
    """
    try:
        # 解析 TLE 数据
        satellite = Satrec.twoline2rv(line1, line2)
    except Exception:
        return False  # 数据格式完全损坏

    # 获取平运动角速度 (sgp4 库中的 no 单位是 弧度/分钟)
    no_rad_min = satellite.no
    if no_rad_min <= 0:
        return False

    # 1. 检查近地点是否在地球内部
    # 转换为 弧度/秒 并计算半长轴 a (km)
    n_rad_sec = no_rad_min / 60.0
    try:
        a = (MU_EARTH / (n_rad_sec**2))**(1/3)
    except ZeroDivisionError:
        return False
    
    # 近地点距离 r_p = a * (1 - e)
    perigee_radius = a * (1 - satellite.ecco)
    if perigee_radius < R_EARTH:
        # 近地点在地球表面以下，轨道不合理（或者已经再入大气层坠毁）
        return False

    # 2. 检查 SGP4 模型是否能成功推演（这里以 TLE 自身的历元时刻进行测试）
    jd = satellite.jdsatepoch
    jdF = satellite.jdsatepochF
    
    # SGP4 库的底层会自动为深空卫星切换到 SDP4 算法
    error_code, r, v = satellite.sgp4(jd, jdF)
    
    if error_code != 0:
        # error_code 不为 0 说明 SGP4 算法报错（例如偏心率>1，轨道衰减等）
        return False

    # 3. 速度异常检查（解决“异常快”的问题）
    # 计算当前位置的模长 (km) 和 速度的模长 (km/s)
    r_mag = math.sqrt(r[0]**2 + r[1]**2 + r[2]**2)
    v_mag = math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
    
    # 计算当前高度的地球逃逸速度 (Escape Velocity)
    # 如果卫星速度大于逃逸速度，说明它不受地球引力束缚，不可能是正常的环地轨道卫星
    v_escape = math.sqrt(2 * MU_EARTH / r_mag)
    
    if v_mag >= v_escape:
        return False
        
    return True

# ================= 测 试 示 例 =================
if __name__ == "__main__":
    # 正常的高轨卫星 (例如北斗/GPS)，速度应该正常
    good_tle_l1 = "1 43226U 18018A   23284.50000000  .00000000  00000-0  00000-0 0  9997"
    good_tle_l2 = "2 43226  55.1054  30.4023 0048123 189.2345 170.5432  2.00543210 3214"
    
    # 构造一个极端的异常 TLE（第一行正常，第二行平运动极大，导致速度极快）
    bad_tle_l1 = "1 99999U 20001A   23284.50000000  .00000000  00000-0  00000-0 0  9997"
    bad_tle_l2 = "2 99999  55.1054  30.4023 0048123 189.2345 170.5432 99.00543210 3214"

    print(f"正常 TLE 判定结果: {is_tle_valid(good_tle_l1, good_tle_l2)}")
    print(f"异常 TLE 判定结果: {is_tle_valid(bad_tle_l1, bad_tle_l2)}")