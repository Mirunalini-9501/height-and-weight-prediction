#IMPORTING THE MODULES
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

#PRINTING THE CONTENTS OF THE DATASET
data = pd.read_csv("/content/height-weight-prediction-dataset.csv")

data =data.drop("Index",axis =1)

#RESHAPING HEIGHT AND WEIGHT VALUES
x = data['Height(Inches)']
x = x.values.reshape(len(x),1)

y = data['Weight(Pounds)']
y = y.values.reshape(-1,1)

#IMPLEMENTING LINEAR REGRESSION 
model = LinearRegression().fit(x,y)
r_sq = model.score(x, y)

#PRINTING THE SLOPE AND INTERCEPT VALUES
model.fit(x,y)
print('intercept:', model.intercept_)
print('slope:',model.coef_)

#PREDICTION OF HEIGHT
data["predicted value"] = model.predict(x)

#ACCURACY OF PREDICTION
accuracy=r2_score(y,model.predict(x))
print("the model accuracy is",accuracy*100,"%")

#PLOTTING THE DATASET VALUES IN GRAPH 
plt.scatter(x,y,color = "black")
plt.plot(x,model.predict(x),color="blue")
plt.title("Height-Weight Prediction")
plt.xlabel("Height(Inches)")
plt.ylabel("Weight(Pounds)")
plt.show()