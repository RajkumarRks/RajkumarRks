import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
data=pd.read_csv("/content/sample_data/Chennai.csv")
data.head()
# 'Price' as the target vector
target = "Price"

# 'Area' as the feature matrix
feature = ["Area"]
X_train = data[feature]
y_train = data[target]
y_mean = y_train.mean()
y_pred_baseline = [y_mean] * len(y_train)


len(y_pred_baseline) == len(y_train)
plt.plot( X_train.values,  y_pred_baseline, color="orange")
plt.scatter(x = X_train, y = y_train)
plt.xlabel("Area")
plt.ylabel("Price")
plt.title("India Chennai: Price vs. Area");
from sklearn.metrics import mean_absolute_error
mae_baseline = mean_absolute_error(y_train, y_pred_baseline)

print("Mean apt price", round(y_mean, 2))
print("Baseline MAE:", round(mae_baseline, 2))

model = LinearRegression()
model.fit(X_train, y_train)
#print(model)
y_pred_training = model.predict(X_train)
y_pred_training[:5]
new_house = np.array([[15]])
predicted_price = model.predict(new_house)
print(predicted_price)
