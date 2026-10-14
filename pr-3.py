import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

df = sns.load_dataset('tips')
df = pd.get_dummies(df, drop_first=True)

X = df.drop(columns=['total_bill'])
y = df['total_bill']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

preds = model.predict(X_test)

print('MSE:', mean_squared_error(y_test, preds))
print('R2:', r2_score(y_test, preds))