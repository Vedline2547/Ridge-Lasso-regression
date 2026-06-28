import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge,Lasso
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
# Load California Housing Dataset
data = fetch_california_housing(as_frame=True)
df = data.frame
# Select features(Median Income) and target(Median House Value)
X = df[["MedInc"]]
y = df["MedHouseVal"]
# Transform features to polynomial features
poly = PolynomialFeatures(degree=2,include_bias=False)
X_poly = poly.fit_transform(X)
# Split training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(X_poly,y,random_state=7,test_size=0.2)
# Ridge regression
ridge_model = Ridge(alpha=1)
ridge_model.fit(X_train,y_train)
ridge_predictions = ridge_model.predict(X_test)
# Lasso regression
lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X_train,y_train)
lasso_predictions = lasso_model.predict(X_test)
# Evaluate Ridge regression
ridge_mse = mean_squared_error(y_test,ridge_predictions)
print("Ridge regression mse: ",ridge_mse)
# Evaluate Lasso Regression
lasso_mse = mean_squared_error(y_test,lasso_predictions)
print("Lasso regression mse: ",lasso_mse)
# Visualize Ridge vs Lasso predictions
plt.figure(figsize=(10,6))
plt.scatter(X_test[:,0],y_test,color="blue",label="Actual data",alpha=0.5)
plt.scatter(X_test[:,0],ridge_predictions,color="green",label="Ridge Predictions",alpha=0.5)
plt.scatter(X_test[:,0],lasso_predictions,color="orange",label="Lasso predictions",alpha=0.5)
plt.title("Ridge vs Lasso regression")
plt.xlabel("Median Income(Transformed)")
plt.ylabel("Median House Value in California")
plt.legend()
plt.show()