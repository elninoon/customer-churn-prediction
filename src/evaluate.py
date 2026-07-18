"""
evaluate.py

统一的模型评估函数，避免每个模型都重复写一遍指标计算代码。
"""

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model(model, X_test, y_test, name: str) -> dict:
    """对已训练好的模型（sklearn Pipeline）在测试集上计算标准指标。

    Parameters
    ----------
    model : 已 fit 好的 sklearn/imblearn Pipeline，需实现 predict / predict_proba
    X_test, y_test : 测试集特征与标签
    name : 模型名称，用于结果表标识

    Returns
    -------
    dict，包含 Model / Accuracy / Precision / Recall / F1 / ROC-AUC
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    result = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob),
    }
    return result


def print_full_report(model, X_test, y_test, name: str) -> None:
    """打印详细的分类报告和混淆矩阵，用于单个模型的深入检查（非批量比较场景）。"""
    y_pred = model.predict(X_test)

    print(f"===== {name} =====")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print()


def results_to_dataframe(results: list[dict]) -> pd.DataFrame:
    """把 evaluate_model 返回的多个 dict 汇总成一张对比表，按 ROC-AUC 降序排列。"""
    df = pd.DataFrame(results)
    return df.sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
