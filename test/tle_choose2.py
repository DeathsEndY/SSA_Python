import math
import os
from sgp4.api import Satrec

# --- 地球物理常数 ---
MU_EARTH = 398600.4418  # 地球标准引力参数 (km^3/s^2)
R_EARTH = 6371.0        # 地球平均半径 (km)

def is_tle_valid(line1: str, line2: str) -> bool:
    """
    根据物理合理性和 SGP4 传播状态筛选 TLE 数据。
    返回 True 表示有效，False 表示异常需剔除。
    """
    try:
        satellite = Satrec.twoline2rv(line1, line2)
    except Exception:
        return False

    no_rad_min = satellite.no
    if no_rad_min <= 0:
        return False

    # 1. 碰撞拦截 (近地点检查)
    n_rad_sec = no_rad_min / 60.0
    try:
        a = (MU_EARTH / (n_rad_sec**2))**(1/3)
    except ZeroDivisionError:
        return False
    
    perigee_radius = a * (1 - satellite.ecco)
    if perigee_radius < R_EARTH:
        return False

    # 2. 算法自检
    jd = satellite.jdsatepoch
    jdF = satellite.jdsatepochF
    error_code, r, v = satellite.sgp4(jd, jdF)
    
    if error_code != 0:
        return False

    # 3. 物理上限拦截 (逃逸速度检查)
    r_mag = math.sqrt(r[0]**2 + r[1]**2 + r[2]**2)
    v_mag = math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
    v_escape = math.sqrt(2 * MU_EARTH / r_mag)
    
    if v_mag >= v_escape:
        return False
        
    return True

def process_tle_file(input_filepath: str, output_filepath: str):
    """
    读取包含批量 TLE 数据的 txt 文件，筛选出有效卫星并写入新文件。
    """
    if not os.path.exists(input_filepath):
        print(f"错误: 找不到文件 {input_filepath}")
        return

    valid_tles = []
    invalid_tles = []

    # 读取文件并去除完全空白的行
    with open(input_filepath, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    name = "UNKNOWN_SATELLITE"
    i = 0
    
    # 稳健解析：基于 Line 1 和 Line 2 的开头字符进行识别
    while i < len(lines):
        # 检查当前行是否是 Line 1，且下一行是否是 Line 2
        if lines[i].startswith('1 ') and (i + 1 < len(lines)) and lines[i+1].startswith('2 '):
            line1 = lines[i]
            line2 = lines[i+1]
            
            # 进行筛选
            if is_tle_valid(line1, line2):
                valid_tles.append((name, line1, line2))
            else:
                invalid_tles.append((name, line1, line2))
                
            # 步进两行（跳过已解析的 Line1 和 Line2）
            i += 2
            # 重置 name，防止名称错位
            name = "UNKNOWN_SATELLITE"
        else:
            # 如果不是 TLE 行，则认为当前行是下一颗卫星的名字
            name = lines[i]
            i += 1

    # 将通过筛选的有效 TLE 写入新文件
    with open(output_filepath, 'w', encoding='utf-8') as f:
        for sat in valid_tles:
            f.write(f"{sat[0]}\n{sat[1]}\n{sat[2]}\n\n")

    # 打印处理结果统计
    print("-" * 30)
    print("批处理完成！结果统计：")
    print(f"总计识别卫星: {len(valid_tles) + len(invalid_tles)}")
    print(f"有效并保留: {len(valid_tles)}")
    print(f"异常并剔除: {len(invalid_tles)}")
    print(f"有效数据已保存至: {output_filepath}")
    print("-" * 30)
    
    # 打印被剔除的卫星名单（方便调试）
    if invalid_tles:
        print("\n被剔除的卫星列表:")
        for invalid_sat in invalid_tles:
            print(f"- {invalid_sat[0]}")

# ================= 执 行 =================
if __name__ == "__main__":
    # 请确保同目录下有一个名为 tle_test.txt 的文件
    input_file = "data/all_tle_15days_0330_0852.txt"
    output_file = "data/tle_filtered.txt"
    
    process_tle_file(input_file, output_file)