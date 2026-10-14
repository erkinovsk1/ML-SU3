import pandas as pd
import seaborn as sns

df = sns.load_dataset('titanic')

print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())