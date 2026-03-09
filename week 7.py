import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

path = "data/day31_seaborn.csv"
df = pd.read_csv(path)

sns.histplot(df, x="income", bins=20)
plt.title("Income histplot")
plt.show()

sns.kdeplot(df["age"], fill=True)
plt.title("Age KDE")
plt.show()

sns.boxplot(x="segment", y="income", data=df)
plt.title("Income by Segment")
plt.show()

sns.countplot(x="segment", data=df)
plt.title("Segment counts")
plt.show()







file_path = "data/day32_relationships.csv"
data = pd.read_csv(file_path)

sns.scatterplot(
    data=data,
    x="feature1",
    y="outcome",
    hue="segment",
    style="priority"
)
plt.title("Relationship between feature1 and outcome")
plt.show()

rel_plot = sns.relplot(
    data=data,
    x="feature1",
    y="outcome",
    hue="segment",
    col="priority",
    kind="scatter"
)
plt.show()




path = "data/day33_corr.csv"
df = pd.read_csv(path)

corr = df.corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation Heatmap")
plt.show()











path = "data/day34_patterns.csv"
df = pd.read_csv(path)

g = sns.relplot(data=df, x="time", y="value", hue="segment", col="season", kind="scatter")
plt.show()

sns.lmplot(data=df, x="time", y="value", hue="segment", col="season")
plt.show()







path = "data/day35_project.csv"
df = pd.read_csv(path)

sns.histplot(df, x="income", bins=20)
plt.title("Income distribution")
plt.show()

sns.boxplot(x="segment", y="income", data=df)
plt.title("Income by segment")
plt.show()

sns.scatterplot(x="age", y="spend", hue="segment", data=df)
plt.title("Age vs spend")
plt.show()

corr = df[["age", "income", "spend"]].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlations")
plt.show()
