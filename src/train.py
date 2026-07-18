"""
train.py

项目唯一的训练入口。命令行直接运行：

    python src/train.py

流程：
    读取原始数据 -> 清洗 -> train/test split -> 构建 preprocessor
    -> 分别训练 Logistic Regression / Random Forest / XGBoost
    -> 计算评估指标 -> 保存对比结果 csv -> 保存最优模型 (joblib)

依赖同目录下的 data_processing.py / features.py / evaluate.py，
避免清洗和评估逻辑在多个 notebook 里重复粘贴。
"""

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from data_processing import load_and_clean_data, split_features_target
from evaluate import evaluate_model, results_to_dataframe
from features import build_preprocessor

RANDOM_STATE = 42


def build_models(preprocessor) -> dict[str, Pipeline]:
    """定义要比较的模型，每个都包在同一个 preprocessor 之后。"""
    return {
        "Logistic Regression": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=300, random_state=RANDOM_STATE
                    ),
                ),
            ]
        ),
        "XGBoost": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    XGBClassifier(
                        n_estimators=300,
                        learning_rate=0.05,
                        max_depth=4,
                        random_state=RANDOM_STATE,
                        eval_metric="logloss",
                    ),
                ),
            ]
        ),
    }


def main(
    raw_data_path: str,
    processed_dir: str,
    reports_dir: str,
    models_dir: str,
) -> None:
    processed_dir = Path(processed_dir)
    reports_dir = Path(reports_dir)
    models_dir = Path(models_dir)
    for d in (processed_dir, reports_dir, models_dir):
        d.mkdir(parents=True, exist_ok=True)

    # 1. 读取 + 清洗
    df = load_and_clean_data(raw_data_path)
    X, y = split_features_target(df)

    # 2. 划分训练/测试集（stratify 保证流失比例一致）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")

    # 保存 processed 数据，方便后续单独分析
    train_out = X_train.copy()
    train_out["Churn"] = y_train
    train_out.to_csv(processed_dir / "train.csv", index=False)

    test_out = X_test.copy()
    test_out["Churn"] = y_test
    test_out.to_csv(processed_dir / "test.csv", index=False)

    # 3. 构建预处理器 + 模型
    preprocessor = build_preprocessor(X_train)
    models = build_models(preprocessor)

    # 4. 训练 + 评估
    results = []
    fitted_models = {}
    for name, pipeline in models.items():
        print(f"训练中: {name} ...")
        pipeline.fit(X_train, y_train)
        fitted_models[name] = pipeline
        results.append(evaluate_model(pipeline, X_test, y_test, name))

    results_df = results_to_dataframe(results)
    print("\n模型对比结果（按 ROC-AUC 降序）:")
    print(results_df)

    # 5. 保存结果表
    results_path = reports_dir / "model_comparison.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\n结果已保存到: {results_path}")

    # 6. 保存表现最好的模型
    best_name = results_df.iloc[0]["Model"]
    best_model = fitted_models[best_name]
    model_path = models_dir / "best_model.joblib"
    joblib.dump(best_model, model_path)
    print(f"最优模型 ({best_name}) 已保存到: {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练客户流失预测模型")
    parser.add_argument(
        "--raw-data",
        default="../data/raw/telco_churn.csv",
        help="原始数据 csv 路径",
    )
    parser.add_argument("--processed-dir", default="../data/processed")
    parser.add_argument("--reports-dir", default="../reports")
    parser.add_argument("--models-dir", default="../models")
    args = parser.parse_args()

    main(args.raw_data, args.processed_dir, args.reports_dir, args.models_dir)
