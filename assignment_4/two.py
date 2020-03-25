import pandas as pd
import numpy
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

data = pd.read_csv('data/insurance.csv')

data.head()

def rmse(y_true, y_pred):
    return numpy.sqrt(mean_squared_error(y_true, y_pred))
    
X = pd.get_dummies(data.drop(['charges'], axis=1), columns=['sex', 'smoker', 'region'])
y = data['charges'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=7)
print(f"train size: {X_train.shape[0]}")
print(f"test size: {X_test.shape[0]}")

model = RandomForestRegressor(random_state=7, n_jobs=-1, n_estimators=200)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"test score is {rmse(y_test, y_pred)}")