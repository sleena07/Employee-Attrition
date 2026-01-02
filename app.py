
import streamlit as st
import pickle
import pandas as pd
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt


# ============================
# Load model artifacts
# ============================
@st.cache_resource
def load_artifacts():
    with open("Model/model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("Model/features.pkl", "rb") as f:
        features = pickle.load(f)

    with open("Model/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
        
    with open("Model/feature_importance.pkl", "rb") as f:
        feature_importance = pickle.load(f)

    return model, features, scaler, feature_importance


model, FEATURES, scaler, feature_importance = load_artifacts()


# ============================
# Load data
# ============================
@st.cache_data
def load_data():
    df = pd.read_csv(
        "/Users/leenasingh/Documents/ML Projects/Employee Attrition Project Coursera/Data/employee_data.csv"
    )
    return df


df_raw = load_data()


# ============================
# Basic preprocessing (shared)
# ============================
df_raw=df_raw[df_raw['Attrition']=="No"]
df_raw=df_raw.drop(columns=['Attrition', 'YearsAtCompany',
    'YearsInCurrentRole',
    'YearsSinceLastPromotion',
    'YearsWithCurrManager',
    'StockOptionLevel'])
df_raw["OverTime"] = df_raw["OverTime"].map({"Yes": 1, "No": 0})


# ============================
# UI dataframe (RAW columns)
# ============================
df_ui = df_raw.copy()


# Age Group
def age_bucket(age):
    if age < 25:
        return "18-24 (Entry)"
    elif age < 35:
        return "25-34 (Early-Mid)"
    elif age < 45:
        return "35-44 (Mid)"
    elif age < 55:
        return "45-54 (Senior)"
    else:
        return "55-60 (Pre-Retirement)"


df_ui["Age_Group"] = df_ui["Age"].apply(age_bucket)


# ============================
# Model dataframe (ENCODED)
# ============================
df_model = pd.get_dummies(
    df_raw,
    columns=[
        "MaritalStatus",
        "Gender",
        "Department",
        "BusinessTravel",
        "EducationField",
        "JobRole",
    ],
    drop_first=False
)


# ============================
# Streamlit UI
# ============================
st.title("Employee Attrition Prediction")


# Sidebar Filters
st.sidebar.header("Employee Filters")

department = st.sidebar.multiselect(
    "Department",
    df_ui["Department"].unique(),
    default=df_ui["Department"].unique(),
)

job_level = st.sidebar.multiselect(
    "Job Level",
    sorted(df_ui["JobLevel"].unique()),
    default=sorted(df_ui["JobLevel"].unique()),
)



age_group = st.sidebar.multiselect(
    "Age Group",
    df_ui["Age_Group"].unique(),
    default=df_ui["Age_Group"].unique(),
)

job_satisfaction = st.sidebar.multiselect(
    "Job Satisfaction",
    sorted(df_ui["JobSatisfaction"].unique()),
    default=sorted(df_ui["JobSatisfaction"].unique()),
)


# ============================
# Apply filters (ONLY on UI DF)
# ============================
df_ui_filtered = df_ui.loc[
    (df_ui["Department"].isin(department))
    & (df_ui["JobLevel"].isin(job_level))
    & (df_ui["Age_Group"].isin(age_group))
    & (df_ui["JobSatisfaction"].isin(job_satisfaction))
]


# ============================
# Align model data with UI rows
# ============================
df_model_filtered = df_model.loc[df_ui_filtered.index]


# Ensure feature alignment
X = df_model_filtered.reindex(columns=FEATURES, fill_value=0)

if scaler is not None:
    X = scaler.transform(X)


# ============================
# Predictions
# ============================
df_ui_filtered["Attrition_Risk"] = model.predict_proba(X)[:, 1]

#Display Key Findings
st.markdown("""
### 📌 Key Findings
Employees who left are typically younger, lower-paid, and early in their careers, with fewer years at the company, in role, and with their manager.
Attrition is higher among those experiencing poor work-life balance, high overtime, and low job, relationship, and involvement satisfaction.
Limited training opportunities and absence of stock options are common among leavers.
Exits are more frequent in entry-level roles, especially Sales Representative positions, and among frequent business travelers.
Employees who left also tend to live farther from work and show higher job mobility (more prior companies).
""")


# ============================
# Metrics
# ============================

total_employees = len(df_ui_filtered)

high_risk_70 = (df_ui_filtered["Attrition_Risk"] > 0.7).sum()
high_risk_90 = (df_ui_filtered["Attrition_Risk"] > 0.9).sum()

pct_70 = (high_risk_70 / total_employees * 100) if total_employees > 0 else 0
pct_90 = (high_risk_90 / total_employees * 100) if total_employees > 0 else 0


st.caption("Percentages are calculated relative to filtered employees.")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Employees", total_employees)

with col2:
    st.metric("High Risk (>70%)", high_risk_70)
    st.caption(f"({pct_70:.1f}%)")

with col3:
    st.metric("Very High Risk (>90%)", high_risk_90)
    st.caption(f"({pct_90:.1f}%)")




# ============================
# Visuals
# ============================
st.subheader("Attrition Risk Distribution")
bins = [0, 0.4, 0.7, 0.9, 1.0]
labels = ["Low (<40%)", "Medium (40–70%)", "High (70–90%)", "Critical (>90%)"]

df_ui_filtered["Risk_Bucket"] = pd.cut(
    df_ui_filtered["Attrition_Risk"],
    bins=bins,
    labels=labels,
    include_lowest=True
)

risk_summary = (
    df_ui_filtered["Risk_Bucket"]
    .value_counts()
    .reindex(labels)
    .fillna(0)
)

total = risk_summary.sum()
risk_pct = (risk_summary / total * 100).round(1)

fig, ax = plt.subplots(figsize=(8,4))

bars = ax.bar(
    risk_summary.index,
    risk_summary.values
)

ax.set_title("Employee Attrition Risk – Tier Summary")
ax.set_ylabel("Number of Employees")
ax.set_xlabel("Risk Tier")

# Add labels on bars
for i, bar in enumerate(bars):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{risk_summary.iloc[i]} ({risk_pct.iloc[i]}%)",
        ha="center",
        va="bottom"
    )

st.pyplot(fig)




st.subheader("Very High-Risk Employees")
st.dataframe(
    df_ui_filtered
        .loc[df_ui_filtered["Attrition_Risk"] > 0.9]
        .sort_values(by="Attrition_Risk", ascending=False)
)

# Get feature importance
feat_imp = feature_importance.set_index('Feature')['Absolute Importance']

#top N slider
top_n = st.slider("Select number of top features", 1, 10, 5)

#Grouping one_hot_encoded columns
feat_imp_grouped = feat_imp.groupby(feat_imp.index.str.rsplit('_').str[0]).sum() 

#top N features
top_features = feat_imp_grouped.sort_values(ascending=False).head(top_n)



# Streamlit title
st.subheader("Top 5 Feature Importances")

# Plot in matplotlib
fig, ax = plt.subplots(figsize=(8,5))
top_features.sort_values(ascending=True).plot(kind='barh', ax=ax, color='skyblue')
ax.set_xlabel("Normalized Importance")
ax.set_ylabel("Features")
ax.set_title(f"Top {top_n} Features by Importance")



# Display in Streamlit
st.pyplot(fig)






