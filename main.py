import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Project folders
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
IMAGE_DIR = os.path.join(BASE_DIR, "images")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

sns.set(style="whitegrid")

# -----------------------------
# Synthetic data generation
# -----------------------------
def generate_synthetic_expense_data(num_records=1000, seed=42):
    np.random.seed(seed)

    categories = {
        "Food": (100, 1200),
        "Travel": (150, 3000),
        "Rent": (6000, 15000),
        "Shopping": (300, 5000),
        "Utilities": (500, 3000),
        "Entertainment": (200, 2500),
        "Healthcare": (300, 4000),
        "Education": (500, 6000),
        "Recharge": (100, 1000),
        "Groceries": (200, 2500)
    }

    category_weights = {
        "Food": 0.18,
        "Travel": 0.08,
        "Rent": 0.05,
        "Shopping": 0.12,
        "Utilities": 0.08,
        "Entertainment": 0.10,
        "Healthcare": 0.07,
        "Education": 0.07,
        "Recharge": 0.08,
        "Groceries": 0.17
    }

    payment_methods = ["Cash", "UPI", "Credit Card", "Debit Card", "Net Banking"]
    payment_weights = [0.12, 0.45, 0.18, 0.15, 0.10]

    descriptions = {
        "Food": ["Restaurant", "Cafe", "Lunch", "Dinner", "Snacks"],
        "Travel": ["Cab", "Train", "Bus", "Flight", "Fuel"],
        "Rent": ["House Rent", "PG Rent", "Room Rent"],
        "Shopping": ["Clothes", "Shoes", "Accessories", "Online Shopping"],
        "Utilities": ["Electricity Bill", "Water Bill", "Internet Bill", "Gas Bill"],
        "Entertainment": ["Movie", "Concert", "Subscription", "Game"],
        "Healthcare": ["Medicine", "Checkup", "Tests", "Clinic Visit"],
        "Education": ["Books", "Course Fee", "Exam Fee", "Stationery"],
        "Recharge": ["Mobile Recharge", "Data Pack", "DTH Recharge"],
        "Groceries": ["Vegetables", "Milk", "Essentials", "Supermarket"]
    }

    dates = pd.date_range(start="2025-01-01", end="2025-12-31", freq="D")
    chosen_categories = np.random.choice(
        list(category_weights.keys()),
        size=num_records,
        p=list(category_weights.values())
    )

    records = []

    for category in chosen_categories:
        date = np.random.choice(dates)
        month = pd.Timestamp(date).month

        low, high = categories[category]
        amount = np.random.randint(low, high + 1)

        # Add realistic seasonal patterns
        if category == "Travel" and month in [5, 6, 11, 12]:
            amount *= 1.35
        if category == "Shopping" and month in [10, 11, 12]:
            amount *= 1.25
        if category == "Entertainment" and month in [4, 5, 12]:
            amount *= 1.20

        payment_method = np.random.choice(payment_methods, p=payment_weights)
        description = np.random.choice(descriptions[category])

        records.append({
            "date": pd.Timestamp(date).strftime("%Y-%m-%d"),
            "category": category,
            "amount": round(amount, 2),
            "payment_method": payment_method,
            "description": description
        })

    # Add fixed monthly rent entries for realism
    for month in range(1, 13):
        rent_date = pd.Timestamp(f"2025-{month:02d}-05")
        rent_amount = np.random.randint(8000, 14001)
        records.append({
            "date": rent_date.strftime("%Y-%m-%d"),
            "category": "Rent",
            "amount": round(rent_amount, 2),
            "payment_method": "Net Banking",
            "description": "Monthly Rent"
        })

    df = pd.DataFrame(records)
    return df

# -----------------------------
# Data cleaning
# -----------------------------
def clean_data(df):
    df = df.copy()

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    # Handle missing values
    df["category"] = df["category"].fillna("Unknown")
    df["payment_method"] = df["payment_method"].fillna("Unknown")
    df["description"] = df["description"].fillna("No Description")

    # Convert date
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Remove invalid dates or amounts
    df = df.dropna(subset=["date"])
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df = df[df["amount"] > 0]

    # Standardize category names
    df["category"] = df["category"].str.strip().str.title()
    df["payment_method"] = df["payment_method"].str.strip().str.title()

    return df

# -----------------------------
# Feature engineering
# -----------------------------
def add_features(df):
    df = df.copy()
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["month_name"] = df["date"].dt.strftime("%b")
    df["weekday"] = df["date"].dt.day_name()

    category_budget = {
        "Food": 12000,
        "Travel": 10000,
        "Rent": 14000,
        "Shopping": 9000,
        "Utilities": 6000,
        "Entertainment": 7000,
        "Healthcare": 5000,
        "Education": 8000,
        "Recharge": 2000,
        "Groceries": 10000
    }

    df["category_budget"] = df["category"].map(category_budget).fillna(5000)
    return df

# -----------------------------
# Analysis functions
# -----------------------------
def category_summary(df):
    summary = (
        df.groupby("category")
        .agg(
            total_spend=("amount", "sum"),
            avg_spend=("amount", "mean"),
            transaction_count=("amount", "count")
        )
        .sort_values(by="total_spend", ascending=False)
        .round(2)
    )
    return summary

def monthly_summary(df):
    summary = (
        df.groupby(["month", "month_name"])
        .agg(
            total_spend=("amount", "sum"),
            avg_spend=("amount", "mean"),
            transaction_count=("amount", "count")
        )
        .reset_index()
        .sort_values(by="month")
        .round(2)
    )
    return summary

def payment_summary(df):
    summary = (
        df.groupby("payment_method")
        .agg(
            total_spend=("amount", "sum"),
            transaction_count=("amount", "count")
        )
        .sort_values(by="total_spend", ascending=False)
        .round(2)
    )
    return summary

def monthly_category_pivot(df):
    pivot = pd.pivot_table(
        df,
        values="amount",
        index="month_name",
        columns="category",
        aggfunc="sum",
        fill_value=0
    )

    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    pivot = pivot.reindex(month_order)
    return pivot

def detect_overspending(df):
    monthly_cat = (
        df.groupby(["month_name", "category"])
        .agg(total_spend=("amount", "sum"), budget=("category_budget", "max"))
        .reset_index()
    )
    monthly_cat["over_budget"] = monthly_cat["total_spend"] > monthly_cat["budget"]
    return monthly_cat

# -----------------------------
# Visualization
# -----------------------------
def create_visualizations(df, cat_summary, mon_summary, pay_summary, pivot):
    # 1. Category-wise spending
    plt.figure(figsize=(10, 6))
    cat_summary["total_spend"].plot(kind="bar")
    plt.title("Category-wise Total Spending")
    plt.xlabel("Category")
    plt.ylabel("Total Spend")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_DIR, "category_spending.png"))
    plt.close()

    # 2. Monthly spending trend
    plt.figure(figsize=(10, 6))
    plt.plot(mon_summary["month_name"], mon_summary["total_spend"], marker="o")
    plt.title("Monthly Spending Trend")
    plt.xlabel("Month")
    plt.ylabel("Total Spend")
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_DIR, "monthly_trend.png"))
    plt.close()

    # 3. Payment method pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(
        pay_summary["total_spend"],
        labels=pay_summary.index,
        autopct="%1.1f%%",
        startangle=140
    )
    plt.title("Spending by Payment Method")
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_DIR, "payment_method_pie.png"))
    plt.close()

    # 4. Spending distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df["amount"], bins=30, kde=True)
    plt.title("Expense Amount Distribution")
    plt.xlabel("Amount")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_DIR, "spending_distribution.png"))
    plt.close()

    # 5. Heatmap
    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot, cmap="YlGnBu", annot=True, fmt=".0f")
    plt.title("Monthly Category Spending Heatmap")
    plt.xlabel("Category")
    plt.ylabel("Month")
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_DIR, "monthly_category_heatmap.png"))
    plt.close()

# -----------------------------
# Insight generation
# -----------------------------
def generate_insights(df, cat_summary, mon_summary, pay_summary, over_budget_df):
    total_spend = round(df["amount"].sum(), 2)
    avg_transaction = round(df["amount"].mean(), 2)

    top_category = cat_summary.index[0]
    top_category_spend = round(cat_summary.iloc[0]["total_spend"], 2)

    highest_month = mon_summary.sort_values(by="total_spend", ascending=False).iloc[0]
    most_used_payment = pay_summary["transaction_count"].idxmax()

    over_budget_cases = over_budget_df[over_budget_df["over_budget"] == True]

    lines = []
    lines.append("Expense Tracker Insights Report")
    lines.append("=" * 40)
    lines.append(f"Total Spending: Rs. {total_spend}")
    lines.append(f"Average Transaction Amount: Rs. {avg_transaction}")
    lines.append(f"Top Spending Category: {top_category} (Rs. {top_category_spend})")
    lines.append(
        f"Highest Spending Month: {highest_month['month_name']} "
        f"(Rs. {round(highest_month['total_spend'], 2)})"
    )
    lines.append(f"Most Used Payment Method: {most_used_payment}")
    lines.append(f"Over-budget category-month cases: {len(over_budget_cases)}")

    if len(over_budget_cases) > 0:
        lines.append("\nExamples of Over-budget Cases:")
        sample_cases = over_budget_cases.head(5)
        for _, row in sample_cases.iterrows():
            lines.append(
                f"- {row['month_name']} | {row['category']} | "
                f"Spend: Rs. {round(row['total_spend'], 2)} | "
                f"Budget: Rs. {round(row['budget'], 2)}"
            )

    with open(os.path.join(OUTPUT_DIR, "insights.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return "\n".join(lines)

# -----------------------------
# Save outputs
# -----------------------------
def save_outputs(df, cat_summary, mon_summary, pay_summary):
    df.to_csv(os.path.join(DATA_DIR, "cleaned_expenses.csv"), index=False)
    cat_summary.to_csv(os.path.join(OUTPUT_DIR, "category_summary.csv"))
    mon_summary.to_csv(os.path.join(OUTPUT_DIR, "monthly_summary.csv"), index=False)
    pay_summary.to_csv(os.path.join(OUTPUT_DIR, "payment_summary.csv"))

# -----------------------------
# Main pipeline
# -----------------------------
def main():
    print("Generating synthetic expense data...")
    df = generate_synthetic_expense_data(num_records=1000)

    raw_path = os.path.join(DATA_DIR, "expenses.csv")
    df.to_csv(raw_path, index=False)
    print(f"Raw dataset saved to: {raw_path}")

    print("Cleaning data...")
    df = clean_data(df)

    print("Adding features...")
    df = add_features(df)

    print("Running analysis...")
    cat_summary = category_summary(df)
    mon_summary = monthly_summary(df)
    pay_summary = payment_summary(df)
    pivot = monthly_category_pivot(df)
    over_budget_df = detect_overspending(df)

    print("Saving outputs...")
    save_outputs(df, cat_summary, mon_summary, pay_summary)

    print("Creating visualizations...")
    create_visualizations(df, cat_summary, mon_summary, pay_summary, pivot)

    print("Generating insights...")
    insights = generate_insights(df, cat_summary, mon_summary, pay_summary, over_budget_df)

    print("\nProject completed successfully!")
    print("\nSample Insights:\n")
    print(insights)

if __name__ == "__main__":
    main()