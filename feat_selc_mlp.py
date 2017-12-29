import numpy as np
from numpy import *
import sys
import os
import xlrd
import pandas as pd
import graphviz 
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.neural_network import MLPClassifier
import timeit
from sklearn.ensemble import ExtraTreesClassifier

data=pd.ExcelFile('wave500k.xls')
edata=data.parse(0)
fedata=np.array(edata)
X=fedata[:,1:]
Y=fedata[:,0]

model = ExtraTreesClassifier()
model.fit(X, Y)
imp_feat=model.feature_importances_
ind = np.argpartition(imp_feat, -5)[-5:]
X_train, X_test, y_train, y_test = train_test_split(X[:,ind], Y, test_size=0.33, random_state=42)


start=timeit.default_timer()
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1,activation='logistic')
clf=clf.fit(X_train,y_train)
pr_lab=clf.predict(X_test)
acc1=(float(sum(pr_lab==y_test))/len(y_test))*100
print acc1
eval_mat=precision_recall_fscore_support(y_test, pr_lab, average='weighted')
eval_mat=np.array(eval_mat)
eval_mat=eval_mat.astype(float)
eval_mat1=eval_mat*100
stop=timeit.default_timer()
time1=stop-start

start=timeit.default_timer()
clf = MLPClassifier(solver='sgd', alpha=0.008,hidden_layer_sizes=(5, 8), random_state=8,activation='identity')
clf=clf.fit(X_train,y_train)
pr_lab=clf.predict(X_test)
acc2=(float(sum(pr_lab==y_test))/len(y_test))*100
print acc2
eval_mat=precision_recall_fscore_support(y_test, pr_lab, average='weighted')
eval_mat=np.array(eval_mat)
eval_mat=eval_mat.astype(float)
eval_mat2=eval_mat*100
stop=timeit.default_timer()
time2=stop-start

start=timeit.default_timer()
clf = MLPClassifier(solver='adam', alpha=0.092,hidden_layer_sizes=(5, 16), random_state=16,activation='tanh')
clf=clf.fit(X_train,y_train)
pr_lab=clf.predict(X_test)
acc3=(float(sum(pr_lab==y_test))/len(y_test))*100
print acc3
eval_mat=precision_recall_fscore_support(y_test, pr_lab, average='weighted')
eval_mat=np.array(eval_mat)
eval_mat=eval_mat.astype(float)
eval_mat3=eval_mat*100
stop=timeit.default_timer()
time3=stop-start

print time1
print time2
print time3
print imp_feat

with open('eval_mat.txt','ab') as f:
 f.write(b'\n')
 np.savetxt(f, eval_mat1,delimiter=',',fmt='%2f', newline='')
 np.savetxt(f, eval_mat2,delimiter=',',fmt='%2f', newline='')
 np.savetxt(f, eval_mat3,delimiter=',',fmt='%2f', newline='')

