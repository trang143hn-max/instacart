from mlxtend.frequent_patterns import apriori, association_rules
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Đường dẫn thư mục
base_path = r"C:\Users\Admin\.cache\kagglehub\datasets\yasserh\instacart-online-grocery-basket-analysis-dataset\versions\1"
# Đọc từng file
orders = pd.read_csv(base_path + r"\orders.csv")
order_products_prior = pd.read_csv(base_path + r"\order_products__prior.csv")
products = pd.read_csv(base_path + r"\products.csv")
aisles = pd.read_csv(base_path + r"\aisles.csv")
departments = pd.read_csv(base_path + r"\departments.csv")

# -------------------------------
# 3. Chỉ merge các cột cần thiết để tiết kiệm RAM
orders_small = orders[['order_id', 'order_dow', 'order_hour_of_day']]
products_small = products[['product_id',
                           'product_name', 'aisle_id', 'department_id']]

df = order_products_prior.merge(products_small, on="product_id", how="left")
df = df.merge(orders_small, on="order_id", how="left")
df = df.merge(aisles, on="aisle_id", how="left")
df = df.merge(departments, on="department_id", how="left")

# -------------------------------
# 4. Kiểm tra dữ liệu thiếu & trùng
print("Missing values per column:\n", df.isnull().sum())
print("Number of duplicated rows:", df.duplicated().sum())

# -------------------------------
# 5. EDA cơ bản
# 5.1 Phân bố đơn hàng theo giờ
plt.figure(figsize=(10, 5))
orders.groupby("order_hour_of_day").size().plot(kind='bar')
plt.title("Số lượng đơn hàng theo giờ")
plt.xlabel("Giờ trong ngày")
plt.ylabel("Số đơn hàng")
plt.show()

# 5.2 Top 10 sản phẩm bán chạy
top_products = df['product_name'].value_counts().head(10)
plt.figure(figsize=(10, 5))
sns.barplot(x=top_products.values, y=top_products.index)
plt.title("Top 10 sản phẩm bán chạy")
plt.xlabel("Số lượng bán")
plt.ylabel("Sản phẩm")
plt.show()

# 5.3 Kích thước giỏ hàng
basket_size = df.groupby("order_id").size()
plt.figure(figsize=(10, 5))
sns.histplot(basket_size, bins=30, kde=False)
plt.title("Phân bố kích thước giỏ hàng")
plt.xlabel("Số sản phẩm trong giỏ")
plt.ylabel("Số lượng đơn hàng")
plt.show()

# 6. Market Basket Analysis - RAM SAFE (top 5 products)
top_n = 5

top_names = (
    df["product_name"]
    .value_counts()
    .head(top_n)
    .index.tolist()
)
print("Top products:", top_names)

df_mb = df[df["product_name"].isin(top_names)].copy()

sample_order_count = 10_000
if len(df_mb["order_id"].unique()) > sample_order_count:
    sample_orders = df_mb["order_id"].drop_duplicates().sample(
        sample_order_count, random_state=42)
    df_mb = df_mb[df_mb["order_id"].isin(sample_orders)]

print("Number of unique orders in sample:", df_mb["order_id"].nunique())

basket = (
    df_mb.groupby(["order_id", "product_name"])
    .size()
    .unstack(fill_value=0)
    .astype(bool)
)

print("Basket shape:", basket.shape)
print("Basket columns:", basket.columns.tolist())

frequent_itemsets = apriori(
    basket,
    min_support=0.03,        # giảm xuống 0.03 để dễ có rule
    use_colnames=True,
    max_len=2,
    low_memory=True
)

print("\n--- Frequent itemsets ---")
print(frequent_itemsets.head(10))

# 100% safe: min_threshold rất thấp, metric = "confidence" hoặc "lift"
rules = association_rules(
    frequent_itemsets,
    metric="lift",
    min_threshold=0.01          # giảm threshold cực thấp để tránh empty
)

# Nếu rules trả dict/string, hoặc kiểu gì cũng cố float
if not rules.empty:
    rules["lift"] = pd.to_numeric(rules["lift"], errors="coerce")
    rules["support"] = pd.to_numeric(rules["support"], errors="coerce")
    rules["confidence"] = pd.to_numeric(rules["confidence"], errors="coerce")

print("\n--- Association rules (top by lift) ---")
if rules.empty:
    print("Không có rule nào dù đã giảm min_support=0.03 và min_threshold=0.01.")
    print("Bạn có thể gửi thêm file/dữ liệu thì mình gợi ý kỹ hơn.")
else:
    # Tạo cột antecedents dạng string để plot
    rules["antecedents_str"] = rules["antecedents"].apply(
        lambda x: ', '.join(sorted(list(x)))
    )

    # 100% an toàn với nlargest: chỉ chạy khi có rule
    top_rules = rules.nlargest(10, "lift")
    print(
        top_rules[["antecedents_str", "consequents",
                   "support", "confidence", "lift"]]
    )

    # Vẽ biểu đồ (nếu có rule)
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=top_rules,
        x="lift",
        y="antecedents_str",
        orient="h"
    )
    plt.title(f"Top 10 association rules (top {top_n} products)")
    plt.xlabel("Lift")
    plt.ylabel("Antecedents")
    plt.tight_layout()
    plt.show()
