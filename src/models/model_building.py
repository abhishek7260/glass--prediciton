import numpy as np
import pandas as pd
import os
import pickle
import yaml
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier


train_data=pd.read_csv("E:\\glass quality prediction\\data\\processed\\processed_train_zscore.csv")
x_train=train_data.drop('Type',axis=1)
y_train=train_data['Type']
encoder=LabelEncoder()

model=RandomForestClassifier(n_estimators=100,max_depth=10)
model.fit(x_train,y_train)
pickle.dump(model,open("results/model.pkl","wb"))
