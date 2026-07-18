"""
data_processing.py

负责：
1. 读取原始数据
2. 数据清洗（TotalCharges 类型转换、缺失值处理）
3. 删除无预测价值的列（customerID）
4. 拆分 X / y

被 train.py 和 notebooks/01_eda.ipynb 共同复用，避免清洗逻辑重复。
"""

from pathlib import Path

import pandas as pd


def load_and_clean_data(path: str | Path) -> pd.DataFrame:
    """读取原始 CSV 并完成基础清洗，返回清洗后的完整 DataFrame（含 Churn 列）。

    Parameters
    ----------
    path : str | Path
        原始数据 csv 路径，例如 "data/raw/telco_churn.csv"

    Returns
    -------
    pd.DataFrame
        清洗后的数据：
        - TotalCharges 转为数值型
        - 丢弃转换失败产生的缺失行
        - 删除 customerID 列
    """
    df = pd.read_csv(path)

    # TotalCharges 原始是字符串，其中含有空格字符串，需要 coerce 成 NaN 再丢弃
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna()

    # customerID 只是唯一编号，没有预测价值
    if "customerID" in df.columns:
        df = df.drop("customerID", axis=1)

    return df.reset_index(drop=True)


def split_features_target(
    df: pd.DataFrame, target_col: str = "Churn"
) -> tuple[pd.DataFrame, pd.Series]:
    """将清洗后的 DataFrame 拆分为特征 X 和二分类标签 y。

    y 会被映射为 {"Yes": 1, "No": 0}。
    """
    X = df.drop(target_col, axis=1)
    y = df[target_col].map({"Yes": 1, "No": 0})

    if y.isnull().any():
        raise ValueError(
            f"'{target_col}' 列中存在无法映射为 0/1 的取值，请检查原始数据。"
        )

    return X, y


if __name__ == "__main__":
    # 简单自测：直接运行本文件可以快速检查清洗是否正常
    df = load_and_clean_data("../data/raw/telco_churn.csv")
    X, y = split_features_target(df)
    print("清洗后数据形状:", df.shape)
    print("X 形状:", X.shape, "y 形状:", y.shape)
    print("y 分布:\n", y.value_counts())
