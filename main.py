import pandas as pd
import os

# -----------------------------
# 1. PATH SETUP
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_PATH = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_PATH = os.path.join(BASE_DIR, "data", "processed")

# Create processed folder if not exists
os.makedirs(PROCESSED_PATH, exist_ok=True)

# -----------------------------
# 2. LOAD DATA
# -----------------------------
orders = pd.read_csv(os.path.join(RAW_PATH, "olist_orders_dataset.csv"))
customers = pd.read_csv(os.path.join(RAW_PATH, "olist_customers_dataset.csv"))
items = pd.read_csv(os.path.join(RAW_PATH, "olist_order_items_dataset.csv"))
payments = pd.read_csv(os.path.join(RAW_PATH, "olist_order_payments_dataset.csv"))
reviews = pd.read_csv(os.path.join(RAW_PATH, "olist_order_reviews_dataset.csv"))

print("✅ Data Loaded Successfully\n")

# -----------------------------
# 3. CLEANING
# -----------------------------
orders.drop_duplicates(inplace=True)
customers.drop_duplicates(inplace=True)
items.drop_duplicates(inplace=True)
payments.drop_duplicates(inplace=True)
reviews.drop_duplicates(inplace=True)

# Convert datetime
orders['order_purchase_timestamp'] = pd.to_datetime(
    orders['order_purchase_timestamp'], errors='coerce'
)

print("✅ Data Cleaning Done\n")

# -----------------------------
# 4. MERGE DATA
# -----------------------------
df = orders.merge(customers, on="customer_id", how="left")
df = df.merge(items, on="order_id", how="left")
df = df.merge(payments, on="order_id", how="left")
df = df.merge(reviews, on="order_id", how="left")

print("✅ Data Merged Successfully\n")

# -----------------------------
# 5. FEATURE ENGINEERING
# -----------------------------
df['total_price'] = df['price'] + df['freight_value']
df['month'] = df['order_purchase_timestamp'].dt.to_period('M')

print("✅ Features Created\n")

# -----------------------------
# 6. ANALYSIS
# -----------------------------

# Monthly revenue
monthly_revenue = df.groupby('month')['total_price'].sum().sort_index()

# Top customers
top_customers = (
    df.groupby('customer_unique_id')['total_price']
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

# Average review
avg_review = df['review_score'].mean()

# Order status count
order_status = df['order_status'].value_counts()

# -----------------------------
# 7. PRINT RESULTS
# -----------------------------
print("===== 📊 KEY INSIGHTS =====\n")

print("1. Monthly Revenue (Top 5):")
print(monthly_revenue.head(), "\n")

print("2. Top 10 Customers:")
print(top_customers, "\n")

print(f"3. Average Review Score: {avg_review:.2f}\n")

print("4. Order Status Distribution:")
print(order_status, "\n")

# -----------------------------
# 8. SAVE FINAL DATA
# -----------------------------
output_file = os.path.join(PROCESSED_PATH, "final_dataset.csv")
df.to_csv(output_file, index=False)

print(f"✅ Final dataset saved at: {output_file}")