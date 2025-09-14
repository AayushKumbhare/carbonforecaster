import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle
data = pd.read_csv('/Users/aayushkumbhare/Desktop/carbon-footprint/data/final.csv')
data = data.dropna()

features = ['Carbon-free energy percentage (CFE%)', 'Renewable energy percentage (RE%)', 'hour', 'day_of_week', 'is_summer', 'is_weekend','month', 'hour_cos', 'hour_sin']
target = 'carbon_intensity'

x = data[features]
y = data[target]

split_idx = int(len(data) * 0.8)

x_train = x.iloc[:split_idx]
x_test = x.iloc[split_idx:]
y_train = y.iloc[:split_idx]
y_test = y.iloc[split_idx:]

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

results_df = pd.DataFrame({
    'datetime': data.iloc[split_idx:]['Datetime'],
    'actual': y_test,
    'predicted': y_pred,
    'error': abs(y_test - y_pred)
})

results_df.to_csv('model_predictions.csv', index=False)


print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
print(f"R2 Score: {r2_score(y_test, y_pred)}")

print(f"Feature Importances: {model.feature_importances_}")

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
