# tlesafety.py
# 加入最近距离筛选/时间窗筛选

import sys
import math
import time
from astropy.time import Time, TimeDelta
from closeApproach import CloseApproach, CElement
from satellite import CSatellite
import numpy as np
from operator import attrgetter
from multiprocessing import Pool
import itertools

GeometryMethod_threshold = 10.0
GeometryMethod_Day = 5.0
GeometryMethod_Num = 0

class TLEINFO: pass
class MININFO:
    def __init__(self, Ti=None, minDis=0.0):
        self.Ti = Ti
        self.minDis = minDis

def readCfg(fname="cfg.txt"):
    global GeometryMethod_threshold, GeometryMethod_Day
    with open(fname,'r') as f:
        for line in f:
            if ':' in line:
                key, val = line.split(':', 1)
                key = key.strip().lower()
                val = val.strip()
                
                if key == "geometrymethod_threshold":
                    GeometryMethod_threshold = float(val)
                elif key == "geometrymethod_day":
                    GeometryMethod_Day = float(val)

def readTarTLE(fname="tartle.txt"):
    vec=[]
    with open(fname,'r') as f:
        lines=[l.rstrip('\n') for l in f]
    i=0
    while i+4 < len(lines):
        t = TLEINFO(); t.str1=lines[i]; t.str2=lines[i+2]; t.str3=lines[i+4]
        vec.append(t); i+=6
    return vec

def geometrical_filter(TarSat, ConSat):
    """几何筛选：根据近地点/远地点判断是否可能相交（判断通过返回 True）。"""
    TarEle = TarSat.GetIniOrb()  # 获取目标卫星的初始轨道状态
    ConEle = ConSat.GetIniOrb()  # 获取干扰卫星的初始轨道状态

    TarRp = TarEle.Calrp()
    TarRa = TarEle.Calra()
    ConRp = ConEle.Calrp()
    ConRa = ConEle.Calra()

    # 如果两个轨道的近远地点没有重叠，则不可能发生碰撞
    if TarRp > ConRa + GeometryMethod_threshold or ConRp > TarRa + GeometryMethod_threshold:
        return False
    return True


def method(TarSat, ConSat):
    # 根据周期计算最近距离，并将距离小于阈值的结果保存在txt中
    global GeometryMethod_Num

    print(f"目标1:{TarSat.GetSatName()}")
    print(f"目标2:{ConSat.GetSatName()}")
    
    # 1. 几何筛选
    if not geometrical_filter(TarSat, ConSat):
        return 1

    TarEle = TarSat.GetIniOrb() # 获取目标卫星的初始轨道状态

    CA=CloseApproach() # 计算两个卫星的接近参数的类
    # 计算在指定天数内的目标卫星的轨道周期数，作为计算的周期
    GeometryMethod_Num = int(GeometryMethod_Day * 86400.0 / TarEle.GetTp())

    for i in range(GeometryMethod_Num):
        resultArg = CA.FindArgTime(TarSat, ConSat, i)
        InterPoint= FindMisDis(TarSat, ConSat, resultArg.Tend)
        InterVec = np.linalg.norm(TarSat.Propagate(resultArg.Tend).GetVel() - ConSat.Propagate(resultArg.Tend).GetVel())
        # m_vecDis.append(InterPoint)
        # print(f"周期 {i+1}/{GeometryMethod_Num}: 时间 {resultArg.Tend.utc.iso}, 最近距离 {InterPoint.minDis:.3f} km")
        if InterPoint.minDis < GeometryMethod_threshold:
            # print(f"Warning: 时间 {resultArg.Tend.utc.iso}  最近距离 {InterPoint.minDis:.3f} km 小于阈值 {GeometryMethod_threshold} km, 可能存在碰撞风险")
            with open('data/safety_output.txt', 'a', encoding='utf-8') as f:
                f.write(f"主目标{TarSat.GetSatName()} ID: {TarSat.GetSatID()}, 次目标{ConSat.GetSatName()} ID: {ConSat.GetSatID()} - 时间 {resultArg.Tend.utc.iso}, 最近距离 {InterPoint.minDis:.3f} km, 相对速度 {InterVec:.3f} km/s\n")

    for i in range(GeometryMethod_Num):
        resultArg = CA.FindDecTime(TarSat, ConSat, i)
        InterPoint= FindMisDis(TarSat, ConSat, resultArg.Tend)
        InterVec = np.linalg.norm(TarSat.Propagate(resultArg.Tend).GetVel() - ConSat.Propagate(resultArg.Tend).GetVel())
        # m_vecDis.append(InterPoint)
        # print(f"周期 {i+1}/{GeometryMethod_Num}: 时间 {resultArg.Tend.utc.iso}, 最近距离 {InterPoint.minDis:.3f} km")
        if InterPoint.minDis < GeometryMethod_threshold:
            # print(f"Warning: 时间 {resultArg.Tend.utc.iso}  最近距离 {InterPoint.minDis:.3f} km 小于阈值 {GeometryMethod_threshold} km, 可能存在碰撞风险")
            with open('data/safety_output.txt', 'a', encoding='utf-8') as f:
                f.write(f"主目标{TarSat.GetSatName()} ID: {TarSat.GetSatID()}, 次目标{ConSat.GetSatName()} ID: {ConSat.GetSatID()} - 时间 {resultArg.Tend.utc.iso}, 最近距离 {InterPoint.minDis:.3f} km, 相对速度 {InterVec:.3f} km/s\n")
    
    # m_vecDis.sort(key=attrgetter('minDis'))
    # res = m_vecDis[0]
    # Tend = res.Ti
    # mindis = res.minDis
    # print("实际星历：%d-%d-%d %d:%d:%.3f  理论最近距离(km)：%.3f  相对速度(km/s)：%.3f" % (
    #     Tend.utc.datetime.year, Tend.utc.datetime.month, Tend.utc.datetime.day,
    #     Tend.utc.datetime.hour, Tend.utc.datetime.minute, Tend.utc.datetime.second + Tend.utc.datetime.microsecond*1e-6,
    #     mindis, np.linalg.norm(TarSat.Propagate(Tend).GetVel() - ConSat.Propagate(Tend).GetVel())))
    print()
    return 0

def FindMisDis(Tar1, Con1, Tf):
    Ttrue = FindRootNewton(Tar1, Con1, Tf, 0.1)
    TarOrbEnd = Tar1.Propagate(Ttrue)
    ConOrbEnd = Con1.Propagate(Ttrue)
    TarConR = ConOrbEnd.GetPos() - TarOrbEnd.GetPos()
    TarConV = ConOrbEnd.GetVel() - TarOrbEnd.GetVel()
    # ttt = math.degrees(math.acos(np.dot(TarConR, TarConV) / (np.linalg.norm(TarConR) * np.linalg.norm(TarConV))))
    epsilon = 1e-4 * np.linalg.norm(TarConR) * np.linalg.norm(TarConV)
    # if abs(np.dot(TarConR, TarConV)) > epsilon:
    #     print("最近轨道交点计算错误, rv点积为", np.dot(TarConR, TarConV),"epsilon为",epsilon)
    # if abs(ttt-90)>0.1:
    #     print("最近轨道交点计算错误")
    tmp = MININFO(Ttrue, np.linalg.norm(TarConR))
    return tmp


def FindRootNewton(Tar1, Con1, Ti0, tol):
    """
    使用牛顿迭代法寻找目标时间点，使得两个卫星的RTN点积为零。
    
    参数:
    - Tar1: 目标卫星对象
    - Con1: 干扰卫星对象
    - Ti0: 初始时间 (astropy.time.Time 对象)
    - tol: 容差，迭代终止条件
    
    返回:
    - Ti: 迭代后的时间点
    """
    h = 0.1  # 用于有限差分的步长
    Ti = Ti0  # 当前迭代时间
    delta = tol + 1  # 初始化误差
    max_iter = 20  # 最大迭代次数
    iter_count = 0  # 迭代计数器

    while abs(delta) > tol and iter_count < max_iter:     
        # 计算函数值和有限差分近似导数
        f = GetRTN(Tar1, Con1, Ti)
        f1 = GetRTN(Tar1, Con1, Ti + TimeDelta(h, format='sec'))
        df = (f1 - f) / h  # 有限差分法计算导数
        # 检查导数是否接近零，避免除零错误
        if abs(df) < 1e-12:
            print("Warning: Derivative is too small, stopping iteration.")
            break
        # 更新迭代值
        delta = f / df
        Ti = Ti - TimeDelta(delta, format='sec')
        iter_count += 1

    # 检查是否达到最大迭代次数
    if iter_count == max_iter:
        print("Warning: Maximum iterations reached, solution may not have converged.")
    
    return Ti

def GetRTN(Tar1, Con1, Ti):
    TarOrbEnd=Tar1.Propagate(Ti)
    ConOrbEnd=Con1.Propagate(Ti)
    TarConR=ConOrbEnd.GetPos()-TarOrbEnd.GetPos()
    TarConV=ConOrbEnd.GetVel()-TarOrbEnd.GetVel()
    return np.dot(TarConV, TarConR)

def CalRTN(TarRV, ConRV):
    rr = TarRV[:3]
    vv = TarRV[3:]
    nn = np.cross(rr,vv)
    tt = np.cross(nn,rr)
    mat = np.column_stack([rr/np.linalg.norm(rr), tt/np.linalg.norm(tt), nn/np.linalg.norm(nn)])
    RVJ2000 = TarRV[:3] - ConRV[:3]
    rtn = np.dot(mat.T, RVJ2000)
    return rtn

def process_satellite_pair(pair):
    """处理单个卫星对"""
    i, j, tles, method = pair
    print(f"处理卫星对: [{i}] {tles[i].str1.strip()} & [{j}] {tles[j].str1.strip()}")
    
    TarSat = CSatellite(tles[i].str1, tles[i].str2, tles[i].str3)
    ConSat = CSatellite(tles[j].str1, tles[j].str2, tles[j].str3)
    result = method(TarSat, ConSat)
    
    return result

def process_all_satellite_pairs():
    """处理所有卫星对"""
    start = time.time()
    tles = readTarTLE(fname="data/tartle.txt")
    num_sats = len(tles)
    num_geo_filter = 0
    pairs = []
    print(f"读取数据成功")
    if num_sats < 2:
        print("Error: 需要至少两个卫星的TLE数据进行碰撞风险评估。请检查tartle.txt文件。")
        sys.exit(1)

    for i in range(num_sats):
        for j in range(i + 1, num_sats):
            pairs.append((i, j, tles, method))
    
    # 并行处理
    with Pool(processes=8) as pool:  # processes可以设置为CPU核心数
        results = pool.map(process_satellite_pair, pairs)
    
    # 统计结果
    num_geo_filter = sum(results)

    end = time.time()
    print(f"共处理 {num_sats} 个目标，共处理 {num_sats*(num_sats-1)//2} 对")
    print(f"几何筛选共有 {num_geo_filter} 对被筛除，占 {num_geo_filter/(num_sats*(num_sats-1)//2)*100:.2f}%")
    print(f"所有目标对处理完成，耗时 {end - start:.2f}s, 平均 {(end - start)/(num_sats*(num_sats-1)//2):.3f}s/次")


def single_satellite_safety():
    """处理单个卫星的安全评估"""
    start = time.time()
    tles = readTarTLE(fname="data/tartle_2.txt")
    if not tles:
        print("Error: 没有读取到任何卫星数据。请检查tartle_2.txt文件。")
        return

    num_sats = len(tles)
    if num_sats < 2:
        print("Error: 需要至少两个卫星的TLE数据进行碰撞风险评估。请检查tartle_2.txt文件。")
        return

    TarSat = CSatellite(tles[0].str1, tles[0].str2, tles[0].str3)
    num_geo_filter = 0
    pair_count = num_sats - 1

    for i in range(1, num_sats):
        ConSat = CSatellite(tles[i].str1, tles[i].str2, tles[i].str3)
        num_geo_filter += method(TarSat, ConSat)

    end = time.time()
    print(f"共处理 {num_sats} 个目标，共处理 {pair_count} 对")
    print(f"几何筛选共有 {num_geo_filter} 对被筛除，占 {num_geo_filter / pair_count * 100:.2f}%")
    print(f"所有目标对处理完成，耗时 {end - start:.2f}s, 平均 {(end - start) / pair_count:.3f}s/次")

if __name__ == "__main__":
    readCfg(fname="data/cfg.txt")
    # process_all_satellite_pairs()
    single_satellite_safety()