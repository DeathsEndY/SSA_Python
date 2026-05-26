import pandas as pd
import numpy as np
from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

def prepare_mock_cdm_data():
    """
    模拟生成多组交会事件的 CDM 时间序列数据。
    实际使用时，请替换为解析真实 JSON/XML 格式的 CDM 数据集。
    """
    np.random.seed(42)
    data = []
    labels = {}
    
    # 模拟 200 个交会事件 (event_id)
    for event_id in range(200):
        # 每个事件在 TCA (Time of Closest Approach) 前收到 3 到 8 条随机数量的 CDM
        num_cdms = np.random.randint(3, 9)
        # 距离交会时间的天数（倒序排列）
        time_to_tca = np.sort(np.random.uniform(0.5, 7.0, num_cdms))[::-1]
        
        for t in time_to_tca:
            data.append({
                'event_id': event_id,
                'time_to_tca': t,
                # 错失距离 (Miss Distance, 米)
                'miss_distance': np.random.uniform(50, 5000) * (t / 7.0), 
                # 协方差矩阵对角线元素 (不确定度大小)
                'sigma_r': np.random.uniform(10, 500),
                # 当前 CDM 预估的碰撞概率 (PoC)
                'cdm_poc': np.random.uniform(1e-8, 1e-4) 
            })
        
        # 模拟最终真实标签：1 代表最终评估为高碰撞风险 (需规避)，0 为低风险
        # 轨道数据通常极度不平衡，此处设置 15% 为高风险
        labels[event_id] = np.random.choice([0, 1], p=[0.85, 0.15])
        
    df_cdm = pd.DataFrame(data)
    y = pd.Series(labels)
    return df_cdm, y

def train_collision_prediction_model():
    print("1. 加载并准备 CDM 时间序列数据...")
    df_cdm, y = prepare_mock_cdm_data()
    
    print("2. 使用 tsfresh 进行时间序列特征工程...")
    # 基于 event_id 分组，根据 time_to_tca 的推移提取统计学特征
    # tsfresh 会将诸如 miss_distance 的收敛趋势、方差、自相关性等转化为独立特征列
    extracted_features = extract_features(
        df_cdm, 
        column_id="event_id", 
        column_sort="time_to_tca",
        n_jobs=0 # 0 表示使用所有可用的 CPU 核心
    )
    
    print("3. 清洗与自动特征选择...")
    # 填充因为特定序列过短而产生的 NaN 值
    impute(extracted_features)
    # 根据 p-value 自动剔除对目标变量 y 无预测价值的冗余特征
    X_filtered = select_features(extracted_features, y)
    print(f"-> 原始特征提取维度: {extracted_features.shape}")
    print(f"-> 有效特征降维后: {X_filtered.shape}")
    
    print("4. 划分训练集和测试集...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_filtered, y, test_size=0.25, stratify=y, random_state=42
    )
    
    print("5. 训练随机森林分类器...")
    # class_weight='balanced' 极其重要，用于应对高风险事件(类别1)稀缺的数据不平衡情况
    rf_model = RandomForestClassifier(
        n_estimators=150, 
        max_depth=10, 
        class_weight='balanced', 
        random_state=42
    )
    rf_model.fit(X_train, y_train)
    
    print("6. 测试集验证与模型评估...")
    y_pred = rf_model.predict(X_test)
    print("\n分类指标报告:")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    train_collision_prediction_model()