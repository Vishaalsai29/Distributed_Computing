
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd

mtcars = pd.read_csv('/mnt/datalake/gamma/mtcars.csv')

mtcars = mtcars.set_index('model')
X = mtcars.drop('mpg', axis=1)
y = mtcars['mpg']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

## Train a regression model
model = LinearRegression()
model.fit(X_train, y_train)

## Predict Y-pred values
y_pred = model.predict(X_test)

## Print MSE
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Save the model
import joblib
joblib.dump(model, '/mnt/datalake/gamma/mtcars_model.pkl')
