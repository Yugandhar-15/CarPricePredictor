import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
import pickle

df = pd.read_csv("car data.csv")

print(df.head())

df = df.drop(['Car_Name'], axis=1)

le = LabelEncoder()

df['Fuel_Type'] = le.fit_transform(df['Fuel_Type'])
df['Seller_Type'] = le.fit_transform(df['Seller_Type'])
df['Transmission'] = le.fit_transform(df['Transmission'])

df['Current_Year'] = 2026
df['No_of_Years'] = df['Current_Year'] - df['Year']

df.drop(['Year', 'Current_Year'], axis=1, inplace=True)

X = df.drop(['Selling_Price'], axis=1)
y = df['Selling_Price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor()

model.fit(X_train, y_train)

predictions = model.predict(X_test)

score = r2_score(y_test, predictions)

print("Accuracy:", score)

pickle.dump(model, open("model.pkl", "wb"))

print("Model saved successfully!")