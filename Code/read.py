import pandas as pd 

df=pd.read_csv(r"data.csv")
print(df)

#Assigning X and Y 


Y=df.price
# print(Y)

X=df.drop(['price'],axis=1)
# print(X)

from sklearn.model_selection import train_test_split
