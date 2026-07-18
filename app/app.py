"""
app.py

客户流失预测 - Streamlit 界面

功能：
1. 侧边栏输入单个客户的特征，实时预测流失概率
2. 展示模型对比结果（Logistic Regression / Random Forest / XGBoost）
3. 展示已保存模型的特征重要性（若模型支持）

运行方式（项目根目录下）：
    streamlit run app.py

依赖：
- models/best_model.joblib   由 src/train.py 训练并保存
- reports/model_comparison.csv  同样由 src/train.py 生成
"""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# ---------- 基础配置 ----------
# app.py 位于 app/ 子目录下，项目根目录（models/、reports/ 所在层级）是上一级
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "best_model.joblib"
REPORT_PATH = BASE_DIR / "reports" / "model_comparison.csv"

st.set_page_config(
    page_title="客户流失预测",
    page_icon="📉",
    layout="wide",
)


# ---------- 缓存加载 ----------
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_comparison_report():
    if not REPORT_PATH.exists():
        return None
    return pd.read_csv(REPORT_PATH)


model = load_model()
report_df = load_comparison_report()


# ---------- 侧边栏：客户特征输入 ----------
def user_input_form() -> pd.DataFrame:
    st.sidebar.header("客户信息输入")

    gender = st.sidebar.selectbox("性别 gender", ["Female", "Male"])
    senior_citizen = st.sidebar.selectbox("是否老年人 SeniorCitizen", [0, 1])
    partner = st.sidebar.selectbox("是否有配偶 Partner", ["Yes", "No"])
    dependents = st.sidebar.selectbox("是否有家属 Dependents", ["Yes", "No"])
    tenure = st.sidebar.slider("使用月数 tenure", 0, 72, 12)

    phone_service = st.sidebar.selectbox("是否有电话服务 PhoneService", ["Yes", "No"])
    multiple_lines = st.sidebar.selectbox(
        "是否多线路 MultipleLines", ["Yes", "No", "No phone service"]
    )
    internet_service = st.sidebar.selectbox(
        "网络服务类型 InternetService", ["DSL", "Fiber optic", "No"]
    )
    online_security = st.sidebar.selectbox(
        "在线安全服务 OnlineSecurity", ["Yes", "No", "No internet service"]
    )
    online_backup = st.sidebar.selectbox(
        "在线备份服务 OnlineBackup", ["Yes", "No", "No internet service"]
    )
    device_protection = st.sidebar.selectbox(
        "设备保护服务 DeviceProtection", ["Yes", "No", "No internet service"]
    )
    tech_support = st.sidebar.selectbox(
        "技术支持服务 TechSupport", ["Yes", "No", "No internet service"]
    )
    streaming_tv = st.sidebar.selectbox(
        "流媒体电视 StreamingTV", ["Yes", "No", "No internet service"]
    )
    streaming_movies = st.sidebar.selectbox(
        "流媒体电影 StreamingMovies", ["Yes", "No", "No internet service"]
    )

    contract = st.sidebar.selectbox(
        "合同类型 Contract", ["Month-to-month", "One year", "Two year"]
    )
    paperless_billing = st.sidebar.selectbox(
        "是否无纸化账单 PaperlessBilling", ["Yes", "No"]
    )
    payment_method = st.sidebar.selectbox(
        "支付方式 PaymentMethod",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
    )

    monthly_charges = st.sidebar.slider("月消费 MonthlyCharges", 0.0, 150.0, 70.0)
    total_charges = st.sidebar.number_input(
        "累计消费 TotalCharges", min_value=0.0, value=float(tenure) * monthly_charges
    )

    data = {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }
    return pd.DataFrame([data])


# ---------- 主页面 ----------
st.title("📉 客户流失预测 Dashboard")
st.caption("Telco Customer Churn Prediction · Logistic Regression / Random Forest / XGBoost")

tab_predict, tab_compare, tab_about = st.tabs(["🔮 单客户预测", "📊 模型对比", "ℹ️ 关于"])

# ----- Tab 1: 单客户预测 -----
with tab_predict:
    if model is None:
        st.error(
            f"未找到已训练模型：{MODEL_PATH}\n\n"
            "请先在项目根目录运行：`python src/train.py` 完成训练。"
        )
    else:
        input_df = user_input_form()

        st.subheader("当前输入的客户特征")
        st.dataframe(input_df, use_container_width=True)

        if st.sidebar.button("🚀 预测流失概率", use_container_width=True):
            proba = model.predict_proba(input_df)[0, 1]
            pred = model.predict(input_df)[0]

            col1, col2 = st.columns([1, 2])
            with col1:
                if pred == 1:
                    st.error("### 预测结果：可能流失 ⚠️")
                else:
                    st.success("### 预测结果：不太可能流失 ✅")
                st.metric("流失概率", f"{proba:.1%}")

            with col2:
                st.progress(min(max(proba, 0.0), 1.0))
                if proba >= 0.7:
                    st.warning("流失风险较高，建议优先介入挽留（如合同升级、专属优惠）。")
                elif proba >= 0.4:
                    st.info("流失风险中等，建议纳入常规关怀名单观察。")
                else:
                    st.info("流失风险较低。")
        else:
            st.info("在左侧填写客户信息后，点击「预测流失概率」按钮查看结果。")

# ----- Tab 2: 模型对比 -----
with tab_compare:
    st.subheader("模型评估指标对比")
    if report_df is None:
        st.warning(
            f"未找到对比结果文件：{REPORT_PATH}\n\n"
            "请先运行 `python src/train.py` 生成 reports/model_comparison.csv。"
        )
    else:
        st.dataframe(
            report_df.style.highlight_max(
                subset=["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
                color="#c6f6d5",
            ),
            use_container_width=True,
        )

        metric = st.selectbox(
            "选择要可视化的指标", ["ROC-AUC", "Accuracy", "Precision", "Recall", "F1"]
        )
        chart_df = report_df.set_index("Model")[[metric]]
        st.bar_chart(chart_df)

# ----- Tab 3: 关于 -----
with tab_about:
    st.markdown(
        """
        ### 关于本项目

        本项目基于 Telco Customer Churn 数据集，比较了三种模型对客户流失的预测能力：

        - **Logistic Regression**：作为可解释性强的基线模型
        - **Random Forest**：集成学习方法，捕捉非线性关系
        - **XGBoost**：梯度提升树，通常在结构化数据上表现最佳

        项目结构遵循「探索 → 特征工程 → 训练 → 部署」的标准流程，
        核心逻辑封装在 `src/` 目录下的模块中（`data_processing.py`、
        `features.py`、`evaluate.py`、`train.py`），本界面仅作为
        最终的交互式展示层。
        """
    )
