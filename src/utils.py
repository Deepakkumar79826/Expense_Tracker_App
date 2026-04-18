import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def ensure_project_folders(base_dir):
    """Create required project folders and return their paths."""
    data_dir = os.path.join(base_dir, "data")
    output_dir = os.path.join(base_dir, "outputs")
    image_dir = os.path.join(base_dir, "images")

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)

    return data_dir, output_dir, image_dir


def generate_synthetic_expense_data(num_records=1000, seed=42):
    """Generate synthetic expense records for one year."""
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
        "Groceries": (200, 2500),
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
        "Groceries": 0.17,
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
        "Groceries": ["Vegetables", "Milk", "Essentials", "Supermarket"],
    }

    dates = pd.date_range(start="2025-01-01", end="2025-12-31", freq="D")
    chosen_categories = np.random.choice(
        list(category_weights.keys()),
        size=num_records,
        p=list(category_weights.values()),
    )

    records = []

    for category in chosen_categories:
        date = np.random.choice(dates)
        month = pd.Timestamp(date).month

        low, high = categories[category]
        amount = np.random.randint(low, high + 1)

        # Add seasonal trends
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
            "description": description,
        })

    # Add fixed monthly rent entries
    for month in range(1, 13):
        rent_date = pd.Timestamp(f"2025-{month:02d}-05")
        rent_amount = np.random.randint(8000, 14001)
        records.append({
            "date": rent_date.strftime("%Y-%m-%d"),
            "category": "Rent",
            "amount": round(rent_amount, 2),
            "payment_method": "Net Banking",
            "description": "Monthly Rent",
        })

    return pd.DataFrame(records)


def clean_data(df):
    """Clean and standardize the expense data."""
    df = df.copy()
    df.drop_duplicates(inplace=True)

    df["category"] = df["category"].fillna("Unknown")
    df["payment_method"] = df["payment_method"].fillna("Unknown")
    df["description"] = df["description"].fillna("No Description")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df = df[df["amount"] > 0]

    df["category"] = df["category"].str.strip().str.title()
    df["payment_method"] = df["payment_method"].str.strip().str.title()

    return df


def add_features(df):
    """Create useful analytical features from the date column."""
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
        "Groceries": 10000,
    }

    df["category_budget"] = df["category"].map(category_budget).fillna(5000)
    return df


def category_summary(df):
    """Return category-level summary."""
    return (
        df.groupby("category")
        .agg(
            total_spend=("amount", "sum"),
            avg_spend=("amount", "mean"),
            transaction_count=("amount", "count"),
        )
        .sort_values(by="total_spend", ascending=False)
        .round(2)
    )


def monthly_summary(df):
    """Return month-wise spending summary."""
    return (
        df.groupby(["month", "month_name"])
        .agg(
            total_spend=("amount", "sum"),
            avg_spend=("amount", "mean"),
            transaction_count=("amount", "count"),
        )
        .reset_index()
        .sort_values(by="month")
        .round(2)
    )


def payment_summary(df):
    """Return payment-method level summary."""
    return (
        df.groupby("payment_method")
        .agg(
            total_spend=("amount", "sum"), transaction_count=("amount", "count")
        )
        .sort_values(by="total_spend", ascending=False)
        .round(2)
    )


def monthly_category_pivot(df):
    """Create a month-category pivot table for heatmap plotting."""
    pivot = pd.pivot_table(
        df,
        values="amount",
        index="month_name",
        columns="category",
        aggfunc="sum",
        fill_value=0,
    )

    month_order = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]
    return pivot.reindex(month_order)


def detect_overspending(df):
    """Compare monthly category spend against assigned budget."""
    monthly_cat = (
        df.groupby(["month_name", "category"])
        .agg(total_spend=("amount", "sum"), budget=("category_budget", "max"))
        .reset_index()
    )
    monthly_cat["over_budget"] = monthly_cat["total_spend"] > monthly_cat["budget"]
    return monthly_cat


def save_outputs(df, cat_summary_df, mon_summary_df, pay_summary_df, data_dir, output_dir):
    """Save cleaned data and summaries to CSV files."""
    df.to_csv(os.path.join(data_dir, "cleaned_expenses.csv"), index=False)
    cat_summary_df.to_csv(os.path.join(output_dir, "category_summary.csv"))
    mon_summary_df.to_csv(os.path.join(output_dir, "monthly_summary.csv"), index=False)
    pay_summary_df.to_csv(os.path.join(output_dir, "payment_summary.csv"))


def create_visualizations(df, cat_summary_df, mon_summary_df, pay_summary_df, pivot_df, image_dir):
    """Generate and save all required charts."""
    sns.set(style="whitegrid")

    # Category-wise spending
    plt.figure(figsize=(10, 6))
    cat_summary_df["total_spend"].plot(kind="bar")
    plt.title("Category-wise Total Spending")
    plt.xlabel("Category")
    plt.ylabel("Total Spend")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(image_dir, "category_spending.png"))
    plt.close()

    # Monthly spending trend
    plt.figure(figsize=(10, 6))
    plt.plot(mon_summary_df["month_name"], mon_summary_df["total_spend"], marker="o")
    plt.title("Monthly Spending Trend")
    plt.xlabel("Month")
    plt.ylabel("Total Spend")
    plt.tight_layout()
    plt.savefig(os.path.join(image_dir, "monthly_trend.png"))
    plt.close()

    # Payment method distribution
    plt.figure(figsize=(8, 8))
    plt.pie(
        pay_summary_df["total_spend"],
        labels=pay_summary_df.index,
        autopct="%1.1f%%",
        startangle=140,
    )
    plt.title("Spending by Payment Method")
    plt.tight_layout()
    plt.savefig(os.path.join(image_dir, "payment_method_pie.png"))
    plt.close()

    # Spending distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df["amount"], bins=30, kde=True)
    plt.title("Expense Amount Distribution")
    plt.xlabel("Amount")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(image_dir, "spending_distribution.png"))
    plt.close()

    # Heatmap
    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot_df, cmap="YlGnBu", annot=True, fmt=".0f")
    plt.title("Monthly Category Spending Heatmap")
    plt.xlabel("Category")
    plt.ylabel("Month")
    plt.tight_layout()
    plt.savefig(os.path.join(image_dir, "monthly_category_heatmap.png"))
    plt.close()


def generate_insights(df, cat_summary_df, mon_summary_df, pay_summary_df, over_budget_df, output_dir):
    """Generate a text-based insight report and save it."""
    total_spend = round(df["amount"].sum(), 2)
    avg_transaction = round(df["amount"].mean(), 2)

    top_category = cat_summary_df.index[0]
    top_category_spend = round(cat_summary_df.iloc[0]["total_spend"], 2)

    highest_month = mon_summary_df.sort_values(by="total_spend", ascending=False).iloc[0]
    most_used_payment = pay_summary_df["transaction_count"].idxmax()
    over_budget_cases = over_budget_df[over_budget_df["over_budget"] == True]

    lines = [
        "Expense Tracker Insights Report",
        "=" * 40,
        f"Total Spending: Rs. {total_spend}",
        f"Average Transaction Amount: Rs. {avg_transaction}",
        f"Top Spending Category: {top_category} (Rs. {top_category_spend})",
        f"Highest Spending Month: {highest_month['month_name']} (Rs. {round(highest_month['total_spend'], 2)})",
        f"Most Used Payment Method: {most_used_payment}",
        f"Over-budget category-month cases: {len(over_budget_cases)}",
    ]

    if len(over_budget_cases) > 0:
        lines.append("\nExamples of Over-budget Cases:")
        for _, row in over_budget_cases.head(5).iterrows():
            lines.append(
                f"- {row['month_name']} | {row['category']} | Spend: Rs. {round(row['total_spend'], 2)} | Budget: Rs. {round(row['budget'], 2)}"
            )

    insights_text = "\n".join(lines)
    with open(os.path.join(output_dir, "insights.txt"), "w", encoding="utf-8") as file:
        file.write(insights_text)

    return insights_text
