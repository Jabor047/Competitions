import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('trainperfsafi.csv')
Y = df['good_bad_flag']
X = df.drop(['good_bad_flag'], axis=1)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.33, random_state=42)

model1 = SVC(gamma='auto')
model3  = LogisticRegression()

model1.fit(X_train,Y_train)
model3.fit(X_train,Y_train)

preds1 = model1.predict(X_test)
preds3 = model3.predict(X_test)

print(accuracy_score(preds1,Y_test))
print(accuracy_score(preds3,Y_test))

