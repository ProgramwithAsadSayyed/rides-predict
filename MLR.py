import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

'''
#this right here is making the model
'''
data = pd.read_csv("taxi.csv", sep=",")

data_x = data.iloc[:, 0:-1].values
data_y = data.iloc[:, -1].values

train_x,test_x,train_y,test_y = train_test_split(data_x,data_y,test_size=0.2,random_state=0)

reg=LinearRegression()

reg.fit(train_x, train_y)
'''
model making finished

NOW WE WILL MAKE A FLASK WEB APP WITH THIS

'''


pickle.dump(reg, open('taxi.pkl', 'wb'))

model=pickle.load(open('taxi.pkl', 'rb'))

print(model.predict([[12,132221,3000,21]]))