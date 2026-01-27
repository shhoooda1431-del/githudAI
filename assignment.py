import numpy as np
import pandas as pd

cat_a = [500, 20000, 10000]
cat_b = [10, 20, 30]
"""___________________________________________________________________________________________________________________________________________________"""
def calculate_state(data):
    mean_1 = sum(data) / len(data)
    max_1 = max(data)
    return {"mean": mean_1, "max": max_1}

print(f"A: {calculate_state(cat_a)} \nB: {calculate_state(cat_b)}")

"""___________________________________________________________________________________________________________________________________________________"""

def calculate_statistics(data):
    return {
        "mean": sum(data) / len(data),
        "max": max(data),
        "min": min(data)
    }

def normalize_data(data, method):
    stats = calculate_statistics(data)
    max__1 = stats["max"]
    min__1 = stats["min"]

    method = method.lower()

    if method == "minmax":
        return [(x - min__1) / (max__1 - min__1) for x in data]

    elif method == "zscore":
        arr = np.array(data)
        mean__1 = np.mean(arr)
        std__1 = np.std(arr)
        return [(x - mean__1) / std__1 for x in data]

def remove_outliers(data, the):
    arr = np.array(data)
    mean__1 = np.mean(arr)
    return [x for x in data if abs(mean__1 - x) <= the]

def test(data,r):
    trine_1=[int(len(data)*r)]
    test_1=[int((r-1.0)*len(data))]
    return test_1 , trine_1
def encode_labels(labels):
    return {label: idx for idx, label in enumerate(set(labels))}
"""____________________________________________________________________________________________________________________________________________________"""

raw = {
    "product": ["Widget A", "Widget B", "Widget C"],
    "price": ["$1,234.50", "$567.89", "$2,345.00"],
    "quantity": [10, 5, None],
}

df = pd.DataFrame(raw)
df["price"] = df["price"].apply(
    lambda x: float(x.replace("$", "").replace(",", ""))
)

df["quantity"] = df["quantity"].fillna(0)
df["total"] = df["price"] * df["quantity"]
def categorize_price(x):
    if x < 50:
        return "low"
    elif x <= 200:
        return "med"
    else:
        return "high"
df["price_category"] = df["price"].apply(categorize_price)
print(df)
    




print(normalize_data(cat_a, "zscore"))
"""print(normalize_data(cat_a,"minmax"))"""

#https://github.com/shhoooda1431-del/githudAI.git
