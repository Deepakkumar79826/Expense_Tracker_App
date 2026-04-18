import os
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "cleaned_expenses.csv")

st.set_page_config(page_title="Expense Tracker Dashboard", layout="wide")
st.title("Expense Tracker Dashboard")

if not os.path.exists(DATA_PATH):
    st.warning("Please run main.py first to generate the cleaned dataset.")
    st.stop()

df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])

st.sidebar.header("Filters")

category_filter = st.sidebar.multiselect(
    "Select Category",
    options=sorted(df["category"].unique()),
    default=sorted(df["category"].unique())
)

payment_filter = st.sidebar.multiselect(
    "Select Payment Method",
    options=sorted(df["payment_method"].unique()),
    default=sorted(df["payment_method"].unique())
)

start_date = st.sidebar.date_input("Start Date", df["date"].min().date())
end_date = st.sidebar.date_input("End Date", df["date"].max().date())

filtered_df = df[
    (df["category"].isin(category_filter)) &
    (df["payment_method"].isin(payment_filter)) &
    (df["date"] >= pd.to_datetime(start_date)) &
    (df["date"] <= pd.to_datetime(end_date))
]

total_spend = round(filtered_df["amount"].sum(), 2)
avg_spend = round(filtered_df["amount"].mean(), 2)
transactions = filtered_df.shape[0]

col1, col2, col3 = st.columns(3)
col1.metric("Total Spend", f"Rs. {total_spend}")
col2.metric("Average Transaction", f"Rs. {avg_spend}")
col3.metric("Transactions", transactions)

st.subheader("Filtered Dataset Preview")
st.dataframe(filtered_df.head(20), use_container_width=True)

st.subheader("Category-wise Spending")
cat_summary = filtered_df.groupby("category")["amount"].sum().sort_values(ascending=False)

fig1, ax1 = plt.subplots(figsize=(8, 4))
cat_summary.plot(kind="bar", ax=ax1)
ax1.set_title("Category-wise Spending")
ax1.set_xlabel("Category")
ax1.set_ylabel("Amount")
plt.xticks(rotation=45)
st.pyplot(fig1)

st.subheader("Monthly Spending Trend")
filtered_df["month_name"] = filtered_df["date"].dt.strftime("%b")
filtered_df["month"] = filtered_df["date"].dt.month

monthly_summary = (
    filtered_df.groupby(["month", "month_name"])["amount"]
    .sum()
    .reset_index()
    .sort_values("month")
)

fig2, ax2 = plt.subplots(figsize=(8, 4))
ax2.plot(monthly_summary["month_name"], monthly_summary["amount"], marker="o")
ax2.set_title("Monthly Spending Trend")
ax2.set_xlabel("Month")
ax2.set_ylabel("Amount")
st.pyplot(fig2)

st.subheader("Payment Method Distribution")
payment_summary = filtered_df.groupby("payment_method")["amount"].sum()

fig3, ax3 = plt.subplots(figsize=(6, 6))
ax3.pie(payment_summary, labels=payment_summary.index, autopct="%1.1f%%", startangle=140)
ax3.set_title("Payment Method Distribution")
st.pyplot(fig3)

st.subheader("Top 5 Highest Expenses")
top5 = filtered_df.sort_values(by="amount", ascending=False).head(5)
st.dataframe(top5[["date", "category", "amount", "payment_method", "description"]], use_container_width=True)