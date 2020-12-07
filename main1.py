#IMPORTING THE MODULES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

#PRINTING THE DATASET
data = pd.DataFrame(data)

#SPLITTING THE GIVEN DATASET FOR PLOTTING THE GRAPH
x = data.iloc[:,0:1]
y = data.iloc[:,1:2]

#IMPLEMENTING POLYNOMIAL REGRESSION
poly_reg = PolynomialFeatures(degree=5)
x_poly = poly_reg.fit_transform(x)

#FITTING POLYNOMIAL REGRESSION ONTO THE LINEAR REGRESSION
poly_reg1 = LinearRegression()
poly_reg1.fit(x_poly,y)

#PLOTTING THE GRAPH USING MATPLOTLIB 
def visualize_polynomial():
  plt.scatter(x,y,color = 'red')
  plt.plot(x,poly_reg1.predict(poly_reg.fit_transform(x)),color = 'blue')
  plt.title("Height and Weight Prediction")
  plt.xlabel("Height(in inches)")
  plt.ylabel("Weight(in pounds)")
  plt.show()
  return
visualize_polynomial()
