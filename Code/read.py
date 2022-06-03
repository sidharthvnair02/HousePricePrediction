import pandas as pd 

df=pd.read_csv(r"single.csv")
print(df)

#Assigning X and Y 


Y=df.price
# print(Y)

X=df.drop(['price'],axis=1)
# print(X)

X_train,X_test,Y_train,Y_test= train_test_split(X,Y, test_size=0.20, random_state=25)
# # print(f"No. of training examples: {training_data.shape[0]}")
# # print(f"No. of testing examples: {testing_data.shape[0]}")
print(X_train.shape,Y_train.shape)

print(X_test.shape,Y_test.shape)

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error , r2_score

#defining the regression model
model=linear_model.LinearRegression()
# df = df.replace(r'^\s*$', np.nan, regex=True)
#building the training model
model.fit(X_train,Y_train)
#applying trained model to make prediction (on test set)
Y_pred = model.predict(X_test)

#print model performance
print('Coefficients:',model.coef_)
print("Intercept:",model.intercept_)
print("Mean squared error:%.2f"% mean_squared_error(Y_test,Y_pred))
print("Coefficients of determination (R^2):%.2f" % r2_score(Y_test,Y_pred))
print(df.head())
Intercept=model.intercept_
sq_feet=float(input("enter sq feet"))
Y=Intercept+280.188*(sq_feet)
print(Y)

import matplotlib.pyplot as plt
#scatter plot 
print(Y_test)
print(Y_pred)
plt.scatter(x=Y_test,y=Y_pred)
plt.title("Y_test v Y_pred")
plt.xlabel("Y_test")
plt.ylabel("Y_predicted")
plt.show()
