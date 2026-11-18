"""
Olist E-Commerce Analysis (Cleaned & Fixed Version)
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)


PATH = r"D:\Olist_Ecommerce_Project"
CHART_DIR = os.path.join(PATH, "charts")
os.makedirs(CHART_DIR, exist_ok=True)

print("Olist E-Commerce Project")
print("Pandas Version:", pd.__version__)
print("NumPy Version:", np.__version__)

print("\nAvailable Dataset Files:")
for f in os.listdir(PATH):
    print(" -", f)


def load(name):
    """Load a CSV from the project folder."""
    return pd.read_csv(os.path.join(PATH, name))


def quick_eda(df, name):
    """Standard EDA summary for any dataframe."""
    print(f"\n{'=' * 60}\n{name.upper()}\n{'=' * 60}")
    print("Shape:", df.shape)
    print("\nFirst 5 Rows:")
    print(df.head())
    print("\nColumn Info:")
    df.info()
    print("\nMissing Values:")
    print(df.isnull().sum())
    print("\nDuplicate Records:", df.duplicated().sum())


def save_show(filename):
    """Save, display and close the current chart."""

    plt.tight_layout()

    plt.savefig(
        os.path.join(CHART_DIR, filename),
        dpi=150,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()



customers = load("olist_customers_dataset.csv")
orders = load("olist_orders_dataset.csv")
order_items = load("olist_order_items_dataset.csv")
products = load("olist_products_dataset.csv")
category = load("product_category_name_translation.csv")
reviews = load("olist_order_reviews_dataset.csv")
payments = load("olist_order_payments_dataset.csv")
sellers = load("olist_sellers_dataset.csv")

# STEP 1: DATA VALIDATION

print("\n========== DATA VALIDATION ==========")

datasets = {
    "Customers": customers,
    "Orders": orders,
    "Order Items": order_items,
    "Products": products,
    "Reviews": reviews,
    "Payments": payments,
    "Sellers": sellers
}

for name, df in datasets.items():

    print(f"\n--- {name} ---")
    print("Rows:", df.shape[0])
    print("Columns:", df.shape[1])
    print("Duplicate Rows:", df.duplicated().sum())
    print("Missing Values:", df.isnull().sum().sum())

    print("Memory Usage:",
          round(df.memory_usage(deep=True).sum() / 1024, 2),
          "KB")

print("\n========== VALIDATION COMPLETED ==========")


# STEP 2: PRIMARY KEY & DATA QUALITY VALIDATION

print("\n========== PRIMARY KEY VALIDATION ==========")

# Unique ID validation
unique_keys = {
    "Customers": (customers, ["customer_id"]),
    "Orders": (orders, ["order_id"]),
    "Products": (products, ["product_id"]),
    "Reviews": (reviews, ["review_id"]),
    "Sellers": (sellers, ["seller_id"])
}

for name, (df, columns) in unique_keys.items():

    print(f"\n--- {name} ---")

    print("Duplicate IDs:",
          df.duplicated(subset=columns).sum())

    print("Missing IDs:",
          df[columns].isnull().sum().sum())


# Composite Key Validation

print("\n--- Order Items ---")

print("Duplicate Order Items:",
      order_items.duplicated(
          subset=["order_id", "order_item_id"]
      ).sum())

print("\n--- Payments ---")

print("Duplicate Payments:",
      payments.duplicated(
          subset=["order_id", "payment_sequential"]
      ).sum())


# NUMERIC VALUE VALIDATION

print("\n========== NUMERIC VALUE CHECK ==========")

print("Negative Product Prices:",
      (order_items["price"] < 0).sum())

print("Negative Freight Values:",
      (order_items["freight_value"] < 0).sum())

print("Invalid Review Scores:",
      (~reviews["review_score"].between(1, 5)).sum())

print("Negative Payment Values:",
      (payments["payment_value"] < 0).sum())

print("\n========== STEP 2 COMPLETED ==========")


# STEP 3: DATA CLEANING

print("\n========== DATA CLEANING STARTED ==========")

# 1. Remove Exact Duplicate Records

datasets = {
    "Customers": customers,
    "Orders": orders,
    "Order Items": order_items,
    "Products": products,
    "Reviews": reviews,
    "Payments": payments,
    "Sellers": sellers
}

for name, df in datasets.items():

    before = len(df)

    df.drop_duplicates(inplace=True)

    after = len(df)

    print(f"\n{name}")
    print("Duplicates Removed:", before - after)


# 2. Remove Missing Primary Keys

customers.dropna(subset=["customer_id"], inplace=True)
orders.dropna(subset=["order_id"], inplace=True)
products.dropna(subset=["product_id"], inplace=True)
reviews.dropna(subset=["review_id"], inplace=True)
sellers.dropna(subset=["seller_id"], inplace=True)


# 3. Remove Invalid Product Prices

invalid_price = (order_items["price"] < 0).sum()

order_items = order_items[
    order_items["price"].isna() |
    (order_items["price"] >= 0)
].copy()

print("\nInvalid Prices Removed:", invalid_price)


# 4. Remove Invalid Freight Values

invalid_freight = (order_items["freight_value"] < 0).sum()

order_items = order_items[
    order_items["freight_value"].isna() |
    (order_items["freight_value"] >= 0)
].copy()

print("Invalid Freight Values Removed:", invalid_freight)


# 5. Remove Invalid Review Scores

invalid_reviews = (
    ~reviews["review_score"].between(1, 5)
    & reviews["review_score"].notna()
).sum()

reviews = reviews[
    reviews["review_score"].isna() |
    reviews["review_score"].between(1, 5)
].copy()

print("Invalid Review Scores Removed:", invalid_reviews)


# 6. Remove Invalid Payment Values

invalid_payments = (payments["payment_value"] < 0).sum()

payments = payments[
    payments["payment_value"].isna() |
    (payments["payment_value"] >= 0)
].copy()

print("Invalid Payment Values Removed:", invalid_payments)


# 7. Final Dataset Shape

print("\n========== CLEANED DATASET SUMMARY ==========")

cleaned_datasets = {
    "Customers": customers,
    "Orders": orders,
    "Order Items": order_items,
    "Products": products,
    "Reviews": reviews,
    "Payments": payments,
    "Sellers": sellers
}

for name, df in cleaned_datasets.items():

    print(f"\n{name}")
    print("Rows:", len(df))
    print("Columns:", len(df.columns))
    print("Missing Values:", df.isnull().sum().sum())
    print("Duplicate Rows:", df.duplicated().sum())


print("\n========== DATA CLEANING COMPLETED ==========")



for df, nm in [
    (customers, "Customers"),
    (orders, "Orders"),
    (order_items, "Order Items"),
    (products, "Products"),
    (reviews, "Reviews"),
    (payments, "Payments"),
    (sellers, "Sellers"),
]:
    quick_eda(df, nm)

print("\nCustomers - Unique Values:")
print(customers.nunique())

# 2. ORDERS: DATE CONVERSION & FEATURES
print("\nOrder Status:")
print(orders["order_status"].value_counts())

date_columns = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]
for col in date_columns:
    orders[col] = pd.to_datetime(orders[col], errors="coerce")

orders["order_year"] = orders["order_purchase_timestamp"].dt.year
orders["order_month"] = orders["order_purchase_timestamp"].dt.month
orders["year_month"] = orders["order_purchase_timestamp"].dt.to_period("M")

# Delivery metrics (only meaningful for delivered orders)
delivered = orders[orders["order_status"] == "delivered"].copy()
delivered = delivered.dropna(subset=["order_delivered_customer_date"])

delivered["delivery_days"] = (
    delivered["order_delivered_customer_date"]
    - delivered["order_purchase_timestamp"]
).dt.days


delivered["delivery_status"] = np.select(
    [
        delivered["order_estimated_delivery_date"].isna(),
        delivered["order_delivered_customer_date"]
        > delivered["order_estimated_delivery_date"]
    ],
    [
        "Unknown",
        "Late"
    ],
    default="On-Time"
)

print("\nDelivery Status:")
print(delivered["delivery_status"].value_counts())
delivered["is_late"] = delivered["delivery_status"] == "Late"

print("\nYearly Orders:")
print(orders["order_year"].value_counts().sort_index())

print("\nMonthly Orders (Year-Month wise):")
monthly_orders = orders.groupby("year_month")["order_id"].nunique()
print(monthly_orders)

print("\nAverage Delivery Days:", round(delivered["delivery_days"].mean(), 2))
print("Late Delivery %:", round(delivered["is_late"].mean() * 100, 2), "%")

# 3. REPEAT CUSTOMER ANALYSIS (delivered orders only)
customer_orders = delivered.merge(
    customers[["customer_id", "customer_unique_id", "customer_state"]],
    on="customer_id",
    how="left",
)

customer_order_count = customer_orders.groupby("customer_unique_id")["order_id"].nunique()
repeat_customers = customer_order_count[customer_order_count > 1]

print("\nTotal Unique Customers:", customer_order_count.count())
print("Repeat Customers:", len(repeat_customers))
print(
    "Repeat Customer Percentage:",
    round(len(repeat_customers) / customer_order_count.count() * 100, 2),
    "%",
)

# 4. ORDER ITEMS: SALES SUMMARY
total_sales = order_items["price"].sum()
total_freight = order_items["freight_value"].sum()

print("\nTotal Product Sales:", round(total_sales, 2))
print("Total Freight Value:", round(total_freight, 2))
print("Freight as % of Sales:", round(total_freight / total_sales * 100, 2), "%")
print("Total Orders:", order_items["order_id"].nunique())
print("Total Items Sold:", len(order_items))

# 5. PRODUCT CATEGORY ANALYSIS
products = products.merge(category, on="product_category_name", how="left")

# Categories missing in the translation file: fall back to the Portuguese
# name, and finally to "unknown".
products["product_category_name_english"] = (
    products["product_category_name_english"]
    .fillna(products["product_category_name"])
    .fillna("unknown")
)

product_sales = order_items.merge(products, on="product_id", how="left")

category_sales = (
    product_sales.groupby("product_category_name_english")["price"]
    .sum()
    .sort_values(ascending=False)
)

print("\nTop 10 Product Categories by Revenue:")
print(category_sales.head(10))

# 6. REVIEWS ANALYSIS
print("\nReview Score Distribution:")
print(reviews["review_score"].value_counts().sort_index())

print("\nAverage Review Score:", round(reviews["review_score"].mean(), 2))

def satisfaction(score):
    if pd.isna(score):
        return "Unknown"
    elif score >= 4:
        return "Satisfied"
    elif score == 3:
        return "Neutral"
    else:
        return "Unsatisfied"

reviews["satisfaction"] = reviews["review_score"].apply(satisfaction)

print("\nCustomer Satisfaction:")
print(reviews["satisfaction"].value_counts())

print("\nSatisfaction Percentage:")
print(reviews["satisfaction"].value_counts(normalize=True).mul(100).round(2))

# Convert Review Timestamp to Datetime

reviews["review_answer_timestamp"] = pd.to_datetime(
    reviews["review_answer_timestamp"],
    errors="coerce"
)
# One review per order (some orders have multiple reviews)
reviews_unique = (
    reviews.sort_values("review_answer_timestamp")
    .drop_duplicates(subset="order_id", keep="last")
)

# Review score vs delivery performance
review_delivery = delivered.merge(
    reviews_unique[["order_id", "review_score"]], on="order_id", how="inner"
)

print("\nAverage Review Score: On-time vs Late")
print(review_delivery.groupby("is_late")["review_score"].mean().round(2))

print("\nCorrelation (delivery_days vs review_score):",
      round(review_delivery["delivery_days"].corr(review_delivery["review_score"]), 3))

# 7. PAYMENT ANALYSIS
print("\nPayment Types:")
print(payments["payment_type"].value_counts())

print("\nPayment Installments:")
print(payments["payment_installments"].value_counts().sort_index())

print("\nTotal Payment Value:", round(payments["payment_value"].sum(), 2))
print("Average Payment Value:", round(payments["payment_value"].mean(), 2))

payment_analysis = (
    payments.groupby("payment_type")["payment_value"]
    .agg(["count", "sum", "mean"])
    .sort_values("sum", ascending=False)
)
print("\nPayment Analysis:")
print(payment_analysis)

# 8. SELLER ANALYSIS
seller_sales = order_items.merge(sellers, on="seller_id", how="left")

seller_performance = (
    seller_sales.groupby("seller_id")
    .agg(
        total_orders=("order_id", "nunique"),
        total_items_sold=("order_item_id", "count"),
        total_revenue=("price", "sum"),
    )
    .sort_values("total_revenue", ascending=False)
)
print("\nTop 10 Sellers by Revenue:")
print(seller_performance.head(10))

seller_location = (
    seller_sales.groupby("seller_state")
    .agg(total_sellers=("seller_id", "nunique"), total_revenue=("price", "sum"))
    .sort_values("total_revenue", ascending=False)
)
print("\nSeller State Performance:")
print(seller_location)

# 9. CUSTOMER STATE ANALYSIS & MONTHLY REVENUE
order_revenue = order_items.groupby("order_id")["price"].sum().reset_index(name="revenue")

state_analysis = (
    orders.merge(customers[["customer_id", "customer_state"]], on="customer_id", how="left")
    .merge(order_revenue, on="order_id", how="left")
    .groupby("customer_state")
    .agg(total_orders=("order_id", "nunique"), total_revenue=("revenue", "sum"))
    .sort_values("total_revenue", ascending=False)
)
print("\nCustomer State Performance (Top 10):")
print(state_analysis.head(10))

monthly_revenue = (
    orders.merge(order_revenue, on="order_id", how="left")
    .groupby("year_month")["revenue"]
    .sum()
)

# 2016 and late-2018 data are incomplete -> use the full-data window for trends
trend_start, trend_end = pd.Period("2017-01"), pd.Period("2018-08")
monthly_orders_trend = monthly_orders.loc[trend_start:trend_end]
monthly_revenue_trend = monthly_revenue.loc[trend_start:trend_end]


# STEP 4: ADVANCED CUSTOMER & MARKETING ANALYSIS

print("\n========== RFM CUSTOMER ANALYSIS ==========")

# 1. Prepare Customer Purchase Data

customer_rfm_data = (
    orders.merge(
        customers[["customer_id", "customer_unique_id"]],
        on="customer_id",
        how="left"
    )
    .merge(
        order_revenue,
        on="order_id",
        how="left"
    )
)

customer_rfm_data = customer_rfm_data.dropna(
    subset=["customer_unique_id", "order_purchase_timestamp"]
)

customer_rfm_data["revenue"] = (
    customer_rfm_data["revenue"].fillna(0)
)

# 2. Calculate RFM Metrics

reference_date = (
    customer_rfm_data["order_purchase_timestamp"].max()
    + pd.Timedelta(days=1)
)

rfm = customer_rfm_data.groupby("customer_unique_id").agg(
    Recency=(
        "order_purchase_timestamp",
        lambda x: (reference_date - x.max()).days
    ),
    Frequency=("order_id", "nunique"),
    Monetary=("revenue", "sum")
)

print("\nRFM Summary:")
print(rfm.describe().round(2))

# 3. Calculate RFM Scores

rfm["R_Score"] = pd.qcut(
    rfm["Recency"].rank(method="first"),
    4,
    labels=[4, 3, 2, 1]
).astype(int)

rfm["F_Score"] = pd.qcut(
    rfm["Frequency"].rank(method="first"),
    4,
    labels=[1, 2, 3, 4]
).astype(int)

rfm["M_Score"] = pd.qcut(
    rfm["Monetary"].rank(method="first"),
    4,
    labels=[1, 2, 3, 4]
).astype(int)

rfm["RFM_Score"] = (
    rfm["R_Score"].astype(str)
    + rfm["F_Score"].astype(str)
    + rfm["M_Score"].astype(str)
)

# 4. Customer Segmentation

def customer_segment(row):

    if row["R_Score"] >= 3 and row["F_Score"] >= 3:
        return "Loyal Customers"

    elif row["R_Score"] >= 3:
        return "Recent Customers"

    elif row["F_Score"] >= 3:
        return "Frequent Customers"

    elif row["R_Score"] <= 2:
        return "At Risk Customers"

    else:
        return "Regular Customers"


rfm["Customer_Segment"] = rfm.apply(
    customer_segment,
    axis=1
)

# 5. Segment Summary

segment_summary = rfm.groupby("Customer_Segment").agg(
    Total_Customers=("Recency", "count"),
    Average_Frequency=("Frequency", "mean"),
    Average_Monetary=("Monetary", "mean")
).round(2)

print("\nCustomer Segment Analysis:")
print(segment_summary)

# 6. Top 10 Customers

top_customers = rfm.sort_values(
    "Monetary",
    ascending=False
).head(10)

print("\nTop 10 Customers by Revenue:")
print(top_customers)

# 7. Customer Segment Visualization

plt.figure(figsize=(10, 6))

segment_summary["Total_Customers"].plot(
    kind="bar"
)

plt.title("Customer Segmentation Analysis")
plt.xlabel("Customer Segment")
plt.ylabel("Number of Customers")
plt.xticks(rotation=30)
plt.grid(axis="y")

save_show("08_customer_segmentation.png")

print("\n========== STEP 4 COMPLETED ==========")


# STEP 5: ADVANCED SALES & PRODUCT ANALYSIS

print("\n========== ADVANCED SALES ANALYSIS ==========")

# 1. Total Revenue

total_revenue = order_items["price"].sum()

print("\nTotal Revenue:", round(total_revenue, 2))


# 2. Average Order Value
order_revenue_series = (
    order_items.groupby("order_id")["price"]
    .sum()
)

average_order_value = order_revenue_series.mean()
print("Average Order Value:", round(average_order_value, 2))


# 3. Total Orders and Products

print("Total Orders:", order_items["order_id"].nunique())

print("Total Products Sold:", len(order_items))

print("Unique Products:",
      order_items["product_id"].nunique())


# 4. Top 10 Products by Revenue

product_performance = (
    order_items.groupby("product_id")
    .agg(
        Total_Sales=("price", "sum"),
        Total_Orders=("order_id", "nunique"),
        Total_Items=("order_item_id", "count")
    )
    .sort_values("Total_Sales", ascending=False)
)

print("\nTop 10 Products by Revenue:")
print(product_performance.head(10))


# 5. Category-wise Revenue

product_category_sales = (
    order_items.merge(
        products[["product_id", "product_category_name_english"]],
        on="product_id",
        how="left"
    )
)

category_performance = (
    product_category_sales.groupby(
        "product_category_name_english"
    )
    .agg(
        Total_Revenue=("price", "sum"),
        Total_Orders=("order_id", "nunique"),
        Total_Items=("product_id", "count")
    )
    .sort_values("Total_Revenue", ascending=False)
)

print("\nTop 10 Categories:")
print(category_performance.head(10))


# 6. Best Selling Products by Quantity

best_selling_products = (
    order_items.groupby("product_id")
    ["order_item_id"]
    .count()
    .sort_values(ascending=False)
)

print("\nTop 10 Best Selling Products:")
print(best_selling_products.head(10))


# 7. Category Revenue Visualization

top_10_categories = (
    category_performance["Total_Revenue"]
    .head(10)
    .sort_values()
)

plt.figure(figsize=(10, 6))

plt.barh(
    top_10_categories.index,
    top_10_categories.values
)

plt.title("Top 10 Categories by Revenue")
plt.xlabel("Total Revenue")
plt.ylabel("Product Category")

save_show("09_advanced_category_revenue.png")


# 8. Top Products Visualization

top_10_products = (
    product_performance["Total_Sales"]
    .head(10)
    .sort_values()
)

plt.figure(figsize=(10, 6))

plt.barh(
    top_10_products.index.astype(str),
    top_10_products.values
)

plt.title("Top 10 Products by Revenue")
plt.xlabel("Total Revenue")
plt.ylabel("Product ID")

save_show("10_top_products.png")


print("\n========== STEP 5 COMPLETED ==========")


# STEP 6: CUSTOMER BEHAVIOUR ANALYSIS

print("\n========== CUSTOMER BEHAVIOUR ANALYSIS ==========")

# 1. Extract Purchase Date Features

purchase_data = orders.copy()

purchase_data = purchase_data.dropna(
    subset=["order_purchase_timestamp"]
)

purchase_data["purchase_day"] = (
    purchase_data["order_purchase_timestamp"].dt.day_name()
)

purchase_data["purchase_hour"] = (
    purchase_data["order_purchase_timestamp"].dt.hour
)

purchase_data["purchase_weekday"] = (
    purchase_data["order_purchase_timestamp"].dt.dayofweek
)

purchase_data["purchase_month"] = (
    purchase_data["order_purchase_timestamp"].dt.month
)

# 2. Day-wise Order Analysis

day_analysis = (
    purchase_data.groupby("purchase_day")["order_id"]
    .nunique()
    .reindex([
        "Monday", "Tuesday", "Wednesday",
        "Thursday", "Friday", "Saturday", "Sunday"
    ])
)

print("\nDay-wise Orders:")
print(day_analysis)

# 3. Hour-wise Purchase Analysis

hour_analysis = (
    purchase_data.groupby("purchase_hour")["order_id"]
    .nunique()
)

print("\nHour-wise Orders:")
print(hour_analysis)

# 4. Weekday vs Weekend

purchase_data["day_type"] = np.where(
    purchase_data["purchase_weekday"] < 5,
    "Weekday",
    "Weekend"
)

day_type_analysis = (
    purchase_data.groupby("day_type")["order_id"]
    .nunique()
)

print("\nWeekday vs Weekend Orders:")
print(day_type_analysis)

# 5. Order Status Analysis

status_analysis = (
    purchase_data["order_status"]
    .value_counts()
)

print("\nOrder Status Analysis:")
print(status_analysis)

# 6. Day-wise Visualization

plt.figure(figsize=(10, 5))

plt.bar(day_analysis.index, day_analysis.values)

plt.title("Customer Purchase by Day")
plt.xlabel("Day")
plt.ylabel("Number of Orders")
plt.xticks(rotation=30)

save_show("11_day_wise_orders.png")

# 7. Hour-wise Visualization

plt.figure(figsize=(10, 5))

plt.plot(
    hour_analysis.index,
    hour_analysis.values,
    marker="o"
)

plt.title("Customer Purchase by Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Number of Orders")
plt.xticks(range(0, 24))
plt.grid(True)

save_show("12_hour_wise_orders.png")

# 8. Weekday vs Weekend Visualization

plt.figure(figsize=(7, 5))

plt.bar(
    day_type_analysis.index,
    day_type_analysis.values
)

plt.title("Weekday vs Weekend Orders")
plt.xlabel("Day Type")
plt.ylabel("Number of Orders")

save_show("13_weekday_weekend_orders.png")

print("\n========== STEP 6 COMPLETED ==========")


# ==========================================
# STEP 7 - DELIVERY PERFORMANCE ANALYSIS
# ==========================================

print("\n========== STEP 7: DELIVERY ANALYSIS ==========")

# 1. Delivery Status Summary

delivery_summary = delivered["delivery_status"].value_counts()

print("\nDelivery Status Summary:")
print(delivery_summary)

# 2. Average Delivery Days

average_delivery_days = delivered["delivery_days"].mean()

print("\nAverage Delivery Days:",
      round(average_delivery_days, 2))

# 3. Late Delivery Percentage

total_delivered = len(delivered)

late_orders = (
    delivered["delivery_status"] == "Late"
).sum()

late_percentage = (
    late_orders / total_delivered * 100
    if total_delivered > 0 else 0
)

print("Total Delivered Orders:", total_delivered)
print("Late Orders:", late_orders)
print("Late Delivery Percentage:",
      round(late_percentage, 2), "%")

# 4. Customer Review vs Delivery Status

delivery_reviews = delivered.merge(
    reviews_unique[["order_id", "review_score"]],
    on="order_id",
    how="left"
)

delivery_reviews = delivery_reviews.dropna(
    subset=["review_score"]
)

review_by_delivery = (
    delivery_reviews[
        delivery_reviews["delivery_status"].isin(
            ["On-Time", "Late"]
        )
    ]
    .groupby("delivery_status")["review_score"]
    .agg(["count", "mean"])
    .round(2)
)

print("\nCustomer Review by Delivery Status:")
print(review_by_delivery)

# 5. Delivery Performance Chart

delivery_order = delivery_summary.reindex(
    ["On-Time", "Late", "Unknown"],
    fill_value=0
)

plt.figure(figsize=(8, 5))

plt.bar(
    delivery_order.index,
    delivery_order.values
)

plt.title("Delivery Performance Analysis")
plt.xlabel("Delivery Status")
plt.ylabel("Number of Orders")

save_show("14_delivery_performance.png")

# 6. Review Comparison Chart

plt.figure(figsize=(8, 5))

review_by_delivery["mean"].plot(kind="bar")

plt.title("Average Review Score by Delivery Status")
plt.xlabel("Delivery Status")
plt.ylabel("Average Review Score")
plt.ylim(0, 5)
plt.xticks(rotation=0)

save_show("15_delivery_review_comparison.png")

print("\n========== STEP 7 COMPLETED ==========")


# STEP 8 - BUSINESS INSIGHTS

print("\n========== STEP 8: BUSINESS INSIGHTS ==========")

# 1. Overall Sales Insights

print("\n--- SALES INSIGHTS ---")

print("Total Revenue:",
      round(order_items["price"].sum(), 2))

print("Total Orders:",
      orders["order_id"].nunique())

print("Total Items Sold:",
      len(order_items))

# 2. Top Product Category

print("\n--- PRODUCT INSIGHTS ---")

if not category_performance.empty:

    top_category = category_performance.index[0]

    top_category_revenue = category_performance.iloc[0][
        "Total_Revenue"
    ]

    print("Top Revenue Category:", top_category)

    print("Category Revenue:",
          round(top_category_revenue, 2))

# 3. Customer Insights

print("\n--- CUSTOMER INSIGHTS ---")

print("Total Unique Customers:",
      customers["customer_unique_id"].nunique())

print("Repeat Customers:",
      len(repeat_customers))

print("Customer Segments:")
print(segment_summary)

# 4. Payment Insights

print("\n--- PAYMENT INSIGHTS ---")

if not payment_analysis.empty:

    top_payment = payment_analysis.index[0]

    print("Most Used Revenue Payment Type:",
          top_payment)

# 5. Delivery Insights

print("\n--- DELIVERY INSIGHTS ---")

print("Average Delivery Days:",
      round(delivered["delivery_days"].mean(), 2))

print("Delivery Status:")
print(delivery_summary)

# 6. Customer Review Insights

print("\n--- CUSTOMER EXPERIENCE ---")

print("Average Review Score:",
      round(reviews["review_score"].mean(), 2))

print("Review by Delivery Status:")
print(review_by_delivery)

# 7. Purchase Behaviour Insights

print("\n--- PURCHASE BEHAVIOUR ---")

print("Most Popular Purchase Day:",
      day_analysis.idxmax())

print("Most Popular Purchase Hour:",
      hour_analysis.idxmax())

print("\n========== STEP 8 COMPLETED ==========")


# STEP 9 - POWER BI DATA EXPORT

print("\n========== STEP 9: POWER BI EXPORT ==========")

# Create Power BI Export Folder
POWERBI_DIR = os.path.join(PATH, "PowerBI_Data")
os.makedirs(POWERBI_DIR, exist_ok=True)

# 1. Orders Summary
powerbi_orders = (
    orders.merge(
        customers[["customer_id", "customer_unique_id", "customer_state"]],
        on="customer_id",
        how="left"
    )
    .merge(
        order_revenue,
        on="order_id",
        how="left"
    )
    .merge(
        delivered[["order_id", "delivery_days", "delivery_status"]],
        on="order_id",
        how="left"
    )
)

powerbi_orders.to_csv(
    os.path.join(POWERBI_DIR, "orders_summary.csv"),
    index=False
)

# 2. Order Items and Product Category
powerbi_order_items = order_items.merge(
    products[["product_id", "product_category_name_english"]],
    on="product_id",
    how="left"
)

powerbi_order_items.to_csv(
    os.path.join(POWERBI_DIR, "order_items_analysis.csv"),
    index=False
)

# 3. Customer RFM Analysis
powerbi_customers = rfm.reset_index()

powerbi_customers.to_csv(
    os.path.join(POWERBI_DIR, "customer_rfm_analysis.csv"),
    index=False
)

# 4. Delivery and Review Analysis
powerbi_delivery = delivery_reviews.copy()

powerbi_delivery.to_csv(
    os.path.join(POWERBI_DIR, "delivery_review_analysis.csv"),
    index=False
)

# 5. Category Performance
powerbi_category = category_performance.reset_index()

powerbi_category.to_csv(
    os.path.join(POWERBI_DIR, "category_performance.csv"),
    index=False
)

# 6. Payment Analysis
powerbi_payment = payment_analysis.reset_index()

powerbi_payment.to_csv(
    os.path.join(POWERBI_DIR, "payment_analysis.csv"),
    index=False
)

# 7. Monthly Sales Analysis
powerbi_monthly = pd.DataFrame({
    "year_month": monthly_revenue.index.astype(str),
    "revenue": monthly_revenue.values,
    "orders": monthly_orders.reindex(
        monthly_revenue.index
    ).values
})

powerbi_monthly.to_csv(
    os.path.join(POWERBI_DIR, "monthly_sales_analysis.csv"),
    index=False
)

print("\nPower BI Export Files:")

for file in os.listdir(POWERBI_DIR):
    print("-", file)

print("\n========== STEP 9 COMPLETED ==========")
print("Export Folder:", POWERBI_DIR)


# 10. VISUALIZATIONS


# 10.1 Monthly Orders
plt.figure(figsize=(12, 5))
plt.plot(monthly_orders_trend.index.astype(str), monthly_orders_trend.values, marker="o")
plt.title("Monthly Orders (Jan 2017 - Aug 2018)")
plt.xlabel("Month")
plt.ylabel("Total Orders")
plt.xticks(rotation=45)
plt.grid(True)
save_show("01_monthly_orders.png")

# 10.2 Monthly Revenue
plt.figure(figsize=(12, 5))
plt.plot(monthly_revenue_trend.index.astype(str), monthly_revenue_trend.values,
         marker="o", color="green")
plt.title("Monthly Revenue (Jan 2017 - Aug 2018)")
plt.xlabel("Month")
plt.ylabel("Revenue")
plt.xticks(rotation=45)
plt.grid(True)
save_show("02_monthly_revenue.png")

# 10.3 Top 10 Categories
top_categories = category_sales.head(10).sort_values()
plt.figure(figsize=(10, 6))
plt.barh(top_categories.index, top_categories.values)
plt.title("Top 10 Product Categories by Revenue")
plt.xlabel("Total Revenue")
plt.ylabel("Product Category")
save_show("03_top_categories.png")

# 10.4 Customer Satisfaction
satisfaction_counts = reviews["satisfaction"].value_counts()
plt.figure(figsize=(7, 7))
plt.pie(satisfaction_counts.values, labels=satisfaction_counts.index,
        autopct="%1.1f%%", startangle=90)
plt.title("Customer Satisfaction Analysis")
save_show("04_satisfaction.png")

# 10.5 Payment Method Revenue
payment_type = (
    payments.groupby("payment_type")["payment_value"].sum().sort_values(ascending=False)
)
plt.figure(figsize=(9, 5))
plt.bar(payment_type.index, payment_type.values)
plt.title("Payment Method Revenue")
plt.xlabel("Payment Type")
plt.ylabel("Total Payment Value")
plt.xticks(rotation=30)
save_show("05_payment_methods.png")

# 10.6 Review Score: On-time vs Late
late_review = review_delivery.groupby("is_late")["review_score"].mean()
plt.figure(figsize=(6, 5))
plt.bar(["On-time", "Late"], [late_review.get(False, 0), late_review.get(True, 0)],
        color=["tab:green", "tab:red"])
plt.title("Average Review Score: On-time vs Late Delivery")
plt.ylabel("Average Review Score")
plt.ylim(0, 5)
save_show("06_review_vs_delivery.png")

# 10.7 Top 10 Customer States by Orders
top_states = state_analysis["total_orders"].head(10)
plt.figure(figsize=(10, 5))
plt.bar(top_states.index, top_states.values)
plt.title("Top 10 Customer States by Orders")
plt.xlabel("State")
plt.ylabel("Total Orders")
save_show("07_top_states.png")

print("\nAnalysis Completed Successfully!")
print("Charts saved in:", CHART_DIR)
