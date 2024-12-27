
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score

# Load dataset
import pandas as pd
file_path = '/path/to/your/dataset.csv'
data = pd.read_csv(file_path)

# Example setup: Splitting into features (X) and target (y)
X = data[['DLength', 'DGirth']]  # Replace with relevant predictors
y = data['SexQ']  # Replace with relevant target variable

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. Simple Linear Regression
simple_model = LinearRegression()
simple_model.fit(X_train[['DLength']], y_train)  # Single feature
y_pred_simple = simple_model.predict(X_test[['DLength']])
print("Simple Linear Regression MSE:", mean_squared_error(y_test, y_pred_simple))

# 2. Multiple Linear Regression
multiple_model = LinearRegression()
multiple_model.fit(X_train, y_train)  # Multiple features
y_pred_multiple = multiple_model.predict(X_test)
print("Multiple Linear Regression MSE:", mean_squared_error(y_test, y_pred_multiple))

# 3. Logistic Regression
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train > 4)  # Example binary classification
y_pred_logistic = logistic_model.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test > 4, y_pred_logistic))

# 4. Ridge Regression
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)
y_pred_ridge = ridge_model.predict(X_test)
print("Ridge Regression MSE:", mean_squared_error(y_test, y_pred_ridge))

# 5. Lasso Regression
lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X_train, y_train)
y_pred_lasso = lasso_model.predict(X_test)
print("Lasso Regression MSE:", mean_squared_error(y_test, y_pred_lasso))
