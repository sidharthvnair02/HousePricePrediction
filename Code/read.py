import pandas as pd 

df=pd.read_csv(r"data.csv")
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
