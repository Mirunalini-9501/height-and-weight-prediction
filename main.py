#IMPORTING THE MODULES
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#PRINTING THE CONTENTS OF THE DATASET
data = pd.read_csv("/content/height-weight-prediction-dataset.csv")
data =data.drop("Index",axis =1)

#IMPLEMENTING THE DATASET
X = data.iloc[:, 1].values  #prints only the height values
y = data.iloc[:, 2].values #prints weight values

#SPLITTING THE DATASET
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=0)


#RESHAPING
X_train= X_train.reshape(-1, 1)
y_train= y_train.reshape(-1, 1)
X_test = X_test.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

#IMPLEMENTING LINEAR REGRESSION 
model = LinearRegression().fit(X_train,y_train)
lin_pred = lin_reg.predict(X_test)

#CALCULATING THE ACCURACY
m = model.score(X_train,y_train)
print("Accuracy is:{}%".format(m*100))

#PLOTTING THE DATASET VALUES IN GRAPH 
plt.scatter(X_train,y_train,color = "red")
plt.plot(X_train,model.predict(X_train),color="blue")
plt.title("Height-Weight Prediction")
plt.xlabel("Height(Inches)")
plt.ylabel("Weight(Pounds)")
plt.show()
