import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv('ecommerce_sales.csv')
df = pd.DataFrame(data)

X = df[['Advertising_Spend', 'Website_Traffic', 'Discount_Rate']]
y = df['Sales']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)


b1, b2, b3 = model.coef_
c = model.intercept_

print(f"Multiple Linear Regression Equation: Sales = {b1:.2f} × Advertising_Spend + {b2:.2f} × Website_Traffic + {b3:.2f} × Discount_Rate + {c:.2f}")


ad_spend = float(input("Enter advertising spend: "))
traffic = int(input("Enter website traffic: "))
discount = float(input("Enter discount rate: "))

user_input_df = pd.DataFrame({
    'Advertising_Spend': [ad_spend],
    'Website_Traffic': [traffic],
    'Discount_Rate': [discount]
})

predicted_sales = model.predict(user_input_df)
print(f"Predicted Sales: {predicted_sales[0]:.2f}")
