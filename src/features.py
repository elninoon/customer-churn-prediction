"""
features.py

负责构建特征预处理器（ColumnTransformer）：
- 数值特征 -> StandardScaler
- 类别特征 -> OneHotEncoder

这个 preprocessor 会被塞进每个模型的 Pipeline 里，
保证 Logistic Regression / Random Forest / XGBoost 用的是同一套特征处理逻辑。
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def get_feature_types(X: pd.DataFrame) -> tuple[pd.Index, pd.Index]:
    """区分数值型和类别型特征列名。

    Returns
    -------
    (numeric_features, categorical_features)
    """
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = X.select_dtypes(include="object").columns
    return numeric_features, categorical_features


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """根据传入的特征 DataFrame 自动构建 ColumnTransformer。

    数值列做标准化，类别列做 One-Hot 编码（未知类别忽略，避免线上出现
    训练集里没见过的取值时报错）。
    """
    numeric_features, categorical_features = get_feature_types(X)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
        ]
    )
    return preprocessor
