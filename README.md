# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
``
1. Import the required library and read the dataframe.
2. Write a function computeCost to generate the cost function.
3. Perform iterations og gradient steps with learning rate.
4. Plot the Cost function using Gradient Descent and generate the required graph.
``
   

## Program:
```
```
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Anumitha.M.R
RegisterNumber: 23007270 
*/
```
```
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate = 0.1, num_iters = 1000):
    X = np.c_[np.ones(len(X1)),X1]
    theta = np.zeros(X.shape[1]).reshape(-1,1)
    
    for _ in range(num_iters):
        predictions = (X).dot(theta).reshape(-1,1)
        errors=(predictions - y ).reshape(-1,1)
        theta -= learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta
data=pd.read_csv("50_Startups.csv")
data.head()
X=(data.iloc[1:,:-2].values)
X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X)
print(X1_Scaled)
theta= linear_regression(X1_Scaled,Y1_Scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(prediction)
print(f"Predicted value: {pre}")
```
``

## Output:

Data Information


![image](https://github.com/anumitha2005/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/155522855/8c5d2b7c-8ef6-4440-b3b6-9ef519b2cc21)


Value of X


![image](https://github.com/anumitha2005/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/155522855/b5c64b3f-be6b-4611-af85-23fdc378570d)


Predicted Value


![image](https://github.com/anumitha2005/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/155522855/8b5121e0-e539-4283-84ce-dad7d3c04075)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
