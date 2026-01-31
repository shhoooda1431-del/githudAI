import numpy as np
import pandas as pd

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
df["total"] = df["price"] * df["quantity"]  #VECTORIZED
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

