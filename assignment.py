import numpy as np
import pandas as pd
import json

cat_a = [500, 20000, 10000]
cat_b = [10, 20, 30]
"""___________________________________ 1 Activity________________________________"""
def calculate_state(data):
    mean_1 = sum(data) / len(data)
    max_1 = max(data)
    return {"mean": mean_1, "max": max_1}

print(f"A: {calculate_state(cat_a)} \nB: {calculate_state(cat_b)}")

"""__________________________ 2 Activity_______________________"""

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
"""_____________________________ 3 Activity_____________________________________"""

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



"""_________________________________ 4 Activity_____________________________________"""
students = {
    "S001": {
        "name": "Alice Chen",
        "courses": {
            "CS101": {"grade": 92, "credits": 3},
            "MATH201": {"grade": 88, "credits": 4},
            "AI301": {"grade": 95, "credits": 3},
        },
        "advisor": "Dr. Smith",
    },
    "S002": {
        "name": "Bob Lee",
        "courses": {
            "CS101": {"grade": 85, "credits": 3},
            "MATH201": {"grade": 90, "credits": 4},
        },
        "advisor": "Dr. Patel",
    },
}

CS101_g=students['S002']['courses']['CS101']['grade']
CS101_c=students['S002']['courses']['CS101']['credits']
MATH201_g=students["S002"]["courses"]["MATH201"]["grade"]
MATH201_c=students["S002"]["courses"]["MATH201"]["credits"]
GPA=(CS101_c*CS101_g + MATH201_c*MATH201_g) / (MATH201_c + CS101_c)

cs101_students = []

for i in students:
    if "CS101" in students[i]["courses"]:
        cs101_students.append(students[i]["name"])



all_grades = []

for sid in students:
    for course in students[sid]["courses"]:
        grade = students[sid]["courses"][course]["grade"]
        all_grades.append(grade)

avg_grade = sum(all_grades) / len(all_grades)



gpa_dict = {}

for sid in students:
    total_points = 0
    total_credits = 0

    for course in students[sid]["courses"]:
        grade = students[sid]["courses"][course]["grade"]
        credits = students[sid]["courses"][course]["credits"]

        total_points += grade * credits
        total_credits += credits

    gpa = total_points / total_credits
    gpa_dict[sid] = gpa


top_student_id = max(gpa_dict, key=gpa_dict.get)
top_student_name = students[top_student_id]["name"]
top_student_gpa = gpa_dict[top_student_id]


print(f"task1 : {students['S001']['courses']['AI301']['grade']} \n task2 : {GPA} \n task3 : {cs101_students} \n task4 : {avg_grade} \n task5 : {top_student_name} with GPA {top_student_gpa}")

"""_________________________________ 5 Activity_____________________________________"""

with open("shahad.txt","w") as f:
  f.write("shahad")
with open("shahad.txt","a") as f:
  f.write("\n shahad12")
"""__________________________________________homework3__________________________________________________"""

df = pd.DataFrame({
    "age": [25, "unknown", 30, "NA", 40],
    "salary": ["5000", "N/A", "?", "7000", "not reported"],
    "city": ["Riyadh", "Jeddah", "unknown", "Dammam", "Riyadh"]
})


tokens = ["N/A", "NA", "not reported", "unknown", "?"]
df = df.replace(tokens, np.nan)


missing_percent = df.isna().mean() * 100


most_harmful = "salary"
reason = "Because salary is numeric and missing values distort its distribution, harming linear and neural models."

print("Cleaned DataFrame:\n", df, "\n")
print("Missing Percentages (%):\n", missing_percent, "\n")
print("Most Harmful Column:", most_harmful)
print("Reason:", reason)

"""_________________________homework4________________________________________"""

raw = {
    "age": [25, "N/A", 40, 33, "?"],
    "income": [50000, 60000, None, "unknown", 80000],
    "churned": [0, 1, 0, 1, 0],
}

df_raw = pd.DataFrame(raw)

missing_tokens = ["N/A", "NA", "not reported", "unknown", "?"]
df = df_raw.replace(missing_tokens, np.nan)


df["age"] = pd.to_numeric(df["age"], errors="coerce")
df["income"] = pd.to_numeric(df["income"], errors="coerce")


missing_per_column = df.isna().mean() * 100
missing_per_row = df.isna().sum(axis=1)


key_cols = ["age", "income"]
df_A = df.dropna(subset=key_cols)

df_B = df.copy()


df_B["age_missing"] = df_B["age"].isna().astype(int)
df_B["income_missing"] = df_B["income"].isna().astype(int)

df_B["age"] = df_B["age"].fillna(df_B["age"].median())
df_B["income"] = df_B["income"].fillna(df_B["income"].median())


comparison = pd.DataFrame({
    "Version A rows": [len(df_A)],
    "Version B rows": [len(df_B)],
    "Age mean A": [df_A["age"].mean()],
    "Age mean B": [df_B["age"].mean()],
    "Income mean A": [df_A["income"].mean()],
    "Income mean B": [df_B["income"].mean()],
})

print("\n=== Cleaned Data ===")
print(df)

print("\n=== Missingness Per Column (%) ===")
print(missing_per_column)

print("\n=== Missingness Per Row ===")
print(missing_per_row)

print("\n=== Version A (Drop Rows) ===")
print(df_A)

print("\n=== Version B (Impute + Indicators) ===")
print(df_B)

print("\n=== Comparison Between A and B ===")
print(comparison)


"""____________________________________________homework5___________________________________"""


train = pd.DataFrame({
    "age": [25, None, 40, 33],
    "city": ["NY", "SF", None, "NY"],
})

test = pd.DataFrame({
    "age": [None, 50],
    "city": ["SF", None],
})


def fit_imputer(train_df, num_cols, cat_cols):
    params = {}
    params["num_medians"] = train_df[num_cols].median()
    params["cat_modes"] = train_df[cat_cols].mode().iloc[0]
    return params

def transform_imputer(df, params, add_indicators=True):
    df2 = df.copy()

 
    for col in params["num_medians"].index:
        if add_indicators:
            df2[col + "_missing"] = df2[col].isna().astype(int)
        df2[col] = df2[col].fillna(params["num_medians"][col])

   
    for col in params["cat_modes"].index:
        if add_indicators:
            df2[col + "_missing"] = df2[col].isna().astype(int)
        df2[col] = df2[col].fillna(params["cat_modes"][col])

    return df2


num_cols = ["age"]
cat_cols = ["city"]

params = fit_imputer(train, num_cols, cat_cols)


train_imputed = transform_imputer(train, params, add_indicators=True)
test_imputed = transform_imputer(test, params, add_indicators=True)

print("=== Imputer Parameters ===")
print(params, "\n")

print("=== Train After Imputation ===")
print(train_imputed, "\n")

print("=== Test After Imputation ===")
print(test_imputed)

"""______________________________________________________________homework6___________________________"""




rows = [
    {"user": "U1", "day": "2024-01-01", "product": "A", "clicked": 1},
    {"user": "U1", "day": "2024-01-01", "product": "A", "clicked": 1},
    {"user": "U1", "day": "2024-01-01", "product": "B", "clicked": 0},
    {"user": "U2", "day": "2024-01-02", "product": "A", "clicked": 1},
]

df = pd.DataFrame(rows)


df_no_exact_dupes = df.drop_duplicates()


df_unique = df_no_exact_dupes.drop_duplicates(subset=["user", "day", "product"])


user_agg = df_unique.groupby("user").agg(
    event_count=("product", "count"),
    ever_clicked=("clicked", "max")
).reset_index()

print(" After Removing Exact Duplicate")
print(df_no_exact_dupes, "\n")

print(" After Applying (user, day, product) Uniqueness ")
print(df_unique, "\n")

print(" User-Level Aggregation ")
print(user_agg)




"""__________Day 10 Activity: Outliers Practice_________________________________"""




np.random.seed(10)
values = np.concatenate([np.random.lognormal(10, 0.5, 1000), [1e7, 2e7]])
df = pd.DataFrame({"income": values})


def iqr_bounds(s: pd.Series, k=1.5):
    Q1 = s.quantile(0.25)
    Q3 = s.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - k * IQR
    upper = Q3 - k * IQR
    return lower, upper

def detect_outliers_iqr(s: pd.Series, k=1.5):
    lower, upper = iqr_bounds(s, k)
    return (s < lower) | (s > upper)



def detect_outliers_zscore(s: pd.Series, threshold=3):
    z = (s - s.mean()) / s.std()
    return z.abs() > threshold

def cap_iqr(s: pd.Series, k=1.5):
    lower, upper = iqr_bounds(s, k)
    return s.clip(lower=lower, upper=upper)



df["income_log1p"] = np.log1p(df["income"])

df["iqr_outlier"] = detect_outliers_iqr(df["income"])
df["z_outlier"] = detect_outliers_zscore(df["income"])
df["income_cap_iqr"] = cap_iqr(df["income"])

summary = pd.DataFrame({
    "original_mean": [df["income"].mean()],
    "capped_mean": [df["income_cap_iqr"].mean()],
    "log1p_mean": [df["income_log1p"].mean()],
    "iqr_outliers_count": [df["iqr_outlier"].sum()],
    "z_outliers_count": [df["z_outlier"].sum()],
})

print("IQR Outliers")
print(df["iqr_outlier"].value_counts(), "\n")

print(" Z‑Score Outliers")
print(df["z_outlier"].value_counts(), "\n")

print("Summary Comparison")
print(summary)


"""_____________________ Day 9 Activity: Data Types Practice Tasks________________"""

import re

raw = {
    "age": ["25", "30", "unknown"],
    "income": ["$50,000", "$60,000", None],
    "signup": ["2024-01-01", "01/05/2024", "not a date"],
}

df = pd.DataFrame(raw)

def normalize_schema(df):
    df2 = df.copy()

    numeric_like = []
    currency_like = []
    datetime_like = []


    for col in df2.columns:
        sample = df2[col].dropna().astype(str).head(5)

        if sample.str.contains(r"^\$|,", regex=True).any():
            currency_like.append(col)
            continue

        if sample.str.contains(r"^\d+(\.\d+)?$", regex=True).any():
            numeric_like.append(col)
            continue

        if sample.str.contains(r"-|/", regex=True).any():
            datetime_like.append(col)
            continue

    for col in currency_like:
        df2[col] = (
            df2[col]
            .astype(str)
            .str.replace(r"[\$,]", "", regex=True)
            .replace("None", np.nan)
        )
        df2[col] = pd.to_numeric(df2[col], errors="coerce")

    for col in numeric_like:
        df2[col] = pd.to_numeric(df2[col], errors="coerce")

    for col in datetime_like:
        df2[col] = pd.to_datetime(df2[col], errors="coerce")

    nan_report = df2.isna().sum()

    return df2, {
        "numeric_like": numeric_like,
        "currency_like": currency_like,
        "datetime_like": datetime_like,
        "nan_counts": nan_report,
    }



df_clean, report = normalize_schema(df)

print("ORIGINAL DATA")
print(df, "\n")

print("CLEANED DATA")
print(df_clean, "\n")

print("REPORT")
print(report)
