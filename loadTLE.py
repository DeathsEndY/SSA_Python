"""
下载tle数据和处理tle数据有关函数
"""

import requests
import os
import sys
import time

def fetch_all_tle(session):
    """
    获取全部在轨目标 TLE (3LE 格式)
    来源: https://www.space-track.org/ (需要注册账号并登录)
    """
    # 输出保存目录 & 保存文件名称
    OUTPUT_DIR = "data"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    FILENAME = "all_tle.txt"

    # Space-Track URL - 获取全部最新在轨目标 TLE（3LE格式）
    URL_ALL_TLE = "https://www.space-track.org/basicspacedata/query/class/gp/decay_date/null-val/format/3le"

    print("Downloading all in-orbit TLE data (3LE format) …")
    resp = session.get(URL_ALL_TLE, stream=True)
    if resp.status_code != 200:
        raise Exception(f"Failed to fetch TLE data: HTTP {resp.status_code}")
    
    # 保存文件
    output_file = os.path.join(OUTPUT_DIR, FILENAME)
    with open(output_file, "w", encoding="utf-8") as f:
        for chunk in resp.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk.decode("utf-8"))
    
    print(f"All TLE data saved to {output_file}")

def fetch_all_tle_days(session, days):
    """
    获取在days天内更新的所有全部在轨目标 TLE (3LE 格式)
    来源: https://www.space-track.org/ (需要注册账号并登录)
    """
    # 输出保存目录 & 保存文件名称
    OUTPUT_DIR = "data"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    FILENAME = f"all_tle_{days}days.txt"

    # Space-Track URL - 获取在days天内更新的最新在轨目标 TLE（3LE格式）
    URL_ALL_TLE = f"https://www.space-track.org/basicspacedata/query/class/gp/decay_date/null-val/epoch/>now-{days}/format/3le"
    
    # URL_ALL_TLE = "https://www.space-track.org/basicspacedata/query/class/gp/decay_date/null-val/epoch/<2026-05-30%2016:10:00/format/3le"

    print("Downloading all in-orbit TLE data (3LE format) …")
    resp = session.get(URL_ALL_TLE, stream=True)
    if resp.status_code != 200:
        raise Exception(f"Failed to fetch TLE data: HTTP {resp.status_code}")
    
    # 保存文件
    output_file = os.path.join(OUTPUT_DIR, FILENAME)
    with open(output_file, "w", encoding="utf-8") as f:
        for chunk in resp.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk.decode("utf-8"))
    
    print(f"All TLE data saved to {output_file}")


def login_session():
    """
    登录space-track网站
    并返回已认证的 requests.Session() 对象
    """
    # 用户账号&密码配置
    USERNAME   = "1697399923@qq.com"
    PASSWORD   = "1697399923thuYJ!"

    # 登录 URL
    LOGIN_URL = "https://www.space-track.org/ajaxauth/login"

    session = requests.Session()
    payload = {
        'identity': USERNAME,
        'password': PASSWORD
    }
    print(f"Logging in as {USERNAME} …")
    resp = session.post(LOGIN_URL, data=payload)
    if resp.status_code != 200:
        raise Exception(f"Login failed: HTTP {resp.status_code}. Response: {resp.text[:200]}")
    print("Login successful.")
    return session

def fetch_active_tle():
    """
    从 CelesTrak 获取所有活跃卫星的 TLE 数据
    来源: https://celestrak.org/ (不需要登陆)
    返回:TLE字符串
    """
    # 输出保存目录 & 保存文件名称
    OUTPUT_DIR = "data"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    FILENAME = "active_tle.txt"

    # CelesTrak URL
    url = f"https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle"

    print(f"Downloading all active satellite TLE data (3LE format)...")

    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        tle_data = response.text

        if tle_data:
            output_file = os.path.join(OUTPUT_DIR, FILENAME)
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(tle_data)
    
            print(f"All TLE data saved to {output_file}")
        else:
            print("no tle_data")

    except requests.RequestException as e:
        print(f"Fail to fetch TLE data: {e}")
        return ""
    
def fetch_single_tle(norad_id):
    """
    根据单个 NORAD 编号从 CelesTrak 获取最新 TLE
    """
    # 构造请求 URL，FORMAT=tle 表示返回标准两行元素格式
    # url = f"https://celestrak.org/NORAD/elements/gp.php?CATNR={norad_id}&FORMAT=tle"
    url = f"https://www.space-track.org/basicspacedata/query/class/gp/NORAD_CAT_ID/{norad_id}/EPOCH/2026-04-20--2026-05-31/format/3le"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status() # 检查请求是否成功
        
        # 返回的是纯文本，按行分割
        lines = response.text.strip().split('\r\n')
        
        # 标准的返回格式是 3 行（名字 + Line1 + Line2）
        if len(lines) == 3:
            name = lines[0].strip()
            line1 = lines[1].strip()
            line2 = lines[2].strip()
            print(f"Downloading TLE data of [{name}] successfully")
            return name, line1, line2
        else:
            print("The satellite was not found or the returned format is incorrect")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"Fail to fetch TLE data: {e}")
        return None
    
def load_tle_from_txt(filepath):
    '''
    从txt中批量提取tle数据为卫星list
    '''
    tle_data = []

    with open(filepath, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    # 每3行一个TLE
    for i in range(0, len(lines), 3):
        name = lines[i]
        line1 = lines[i+1]
        line2 = lines[i+2]

        sat_dict = {
            "name": name,
            "line1": line1,
            "line2": line2
        }

        tle_data.append(sat_dict)

    return tle_data


def main():

    print("请选择获取方式：")
    print("1 - 批量获取全部轨道目标的TLE")
    print("2 - 批量获取在days天内更新的全部轨道目标的TLE")
    print("3 - 批量获取在轨存活卫星的TLE")
    print("4 - 获取指定 NORAD ID 的 TLE")

    choice = input("请输入 1/2/3/4: ").strip()
    
    if choice == "1":
        try:
            session = login_session()
        except Exception as e:
            print(f"Login error: {e}", file=sys.stderr)
            return

        try:
            fetch_all_tle(session)
        except Exception as e:
            print(f"Error fetching all TLE: {e}", file=sys.stderr)
        finally:
            session.close()
        print("Done.")

    elif choice == "2":
        try:
            session = login_session()
        except Exception as e:
            print(f"Login error: {e}", file=sys.stderr)
            return

        days_input = input("请输入整数天数 (如10): ").strip()
        try:
            days = int(days_input)
        except ValueError:
            print("输入格式错误，请输入整数天数")
            return

        try:
            fetch_all_tle_days(session, days)
        except Exception as e:
            print(f"Error fetching TLE for last {days} days: {e}", file=sys.stderr)
        finally:
            session.close()
        print("Done.")

    elif choice == "3":
        try:
            fetch_active_tle()
        except Exception as e:
            print(f"Error fetching active satellite TLE: {e}", file=sys.stderr)
        print("Done.")

    elif choice == "4":

        ids_input = input("请输入 NORAD ID(多个用空格分隔): ").strip()  #如48274 68110 68103 68115 68116 68104 68106

        try:
            norad_ids = [int(x) for x in ids_input.split()]
        except ValueError:
            print("输入格式错误，请输入整数 ID")
            return

        output_file = "data/part_tle.txt"

        with open(output_file, "w", encoding="utf-8") as f:
            for norad_id in norad_ids:
                result = fetch_single_tle(norad_id)
                if result:
                    name, line1, line2 = result
                    f.write(name + "\n\n")
                    f.write(line1 + "\n\n")
                    f.write(line2 + "\n\n")
                else:
                    print(f"未获取到 NORAD ID={norad_id} 的数据")

        print(f"TLE 已保存到 {output_file}")

    else:
        print("无效输入，请输入 1/2/3")


if __name__ == "__main__":
    main()