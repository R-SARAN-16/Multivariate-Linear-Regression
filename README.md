# Implementation of Multivariate Linear Regression
## Aim
To write a python program to implement multivariate linear regression and predict the output.
## Equipment’s required:
1.	Hardware – PCs
2.	Anaconda – Python 3.7 Installation / Moodle-Code Runner
## Algorithm:
### Step1
import numpy,matplotlib,datasets,linear_model,metrices

### Step2
Define a feature matrix and response vector

### Step3
Create a linear regression object

### Step4
Train the model using training sets

### Step5
Plot for residual error

## Program:
```python

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model

# Load the Diabetes dataset
diabetes = datasets.load_diabetes()

# Define feature matrix (X) and response vector (y)
X = diabetes.data
y = diabetes.target

# Split data into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=1
)

# Create linear regression object
reg = linear_model.LinearRegression()

# Train the model
reg.fit(X_train, y_train)

# Display regression coefficients
print("Coefficients:", reg.coef_)

# Display variance score
print("Variance score: {}".format(reg.score(X_test, y_test)))

# Plot residual errors
plt.style.use('fivethirtyeight')

# Training data residuals
plt.scatter(
    reg.predict(X_train),
    reg.predict(X_train) - y_train,
    color="green", s=10, label="Train data"
)

# Testing data residuals
plt.scatter(
    reg.predict(X_test),
    reg.predict(X_test) - y_test,
    color="blue", s=10, label="Test data"
)

# Zero residual error line
plt.hlines(
    y=0,
    xmin=min(reg.predict(X_test)),
    xmax=max(reg.predict(X_test)),
    linewidth=2
)

# Display legend and title
plt.legend(loc="upper right")
plt.title("Residual Errors")
plt.xlabel("Predicted Values")
plt.ylabel("Residual Errors")

plt.savefig("residual_errors.png")
plt.show()

```
## Output:

<img width="837" height="692" alt="image" src="https://github.com/user-attachments/assets/53a9ee21-0ba4-4d00-970f-ca0cf7a2052e" />

<br>

## Result
Thus the multivariate linear regression is implemented and predicted the output using python program.
