from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor

data = fetch_california_housing()
X = data.data

y = data.target

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

x1 = [0] * X.shape[1]
x2 = [1] * X.shape[1]

p1 = model.predict([x1])[0]
p2 = model.predict([x2])[0]

print('p1', p1)
print('p2', p2)

