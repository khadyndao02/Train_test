import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy import optimize
from scipy import signal
from scipy import fftpack
from scipy import ndimage
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split #pour diviser le dataset
from sklearn.model_selection import cross_val_score #pour valider le score
from sklearn.model_selection import validation_curve
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix

import time

# train test split
#Quand on fait du machine learning on divise la dataset en deux parties:
#-un train set dont les données sont utilisées pour entrainer le modéle (80%)
#-un test set reserver uniquement à l'évaluation du modele (20%)
# On utilise le model.fit(Xtrain,Ytrain)pour entrainer le model
# On utilise le model.score(Xtest,Ytest)pour évaluer le model

iris =load_iris()
print(iris)
X = iris.data
y = iris.target
print(X.shape)
plt.scatter(X[:,0],X[:,1],c=y,alpha = 0.8)
plt.show()
X_train,X_test,y_train, y_test = train_test_split(X,y,test_size=0.2,random_state = 5)
print('Train set : ', X_train.shape)
print('Test set : ', X_test.shape)
plt.figure(figsize =(12,4))
plt.subplot(121)
plt.scatter(X_train[:,0],X_train[:,1], c=y_train, alpha = 0.8)
plt.title('Train set')
plt.subplot(122)
plt.show()
plt.scatter(X_test[:,0],X_test[:,1], c=y_test, alpha = 0.8)
plt.title('Test set')
plt.show()

model = KNeighborsClassifier(n_neighbors = 1)
print(model.fit(X_train,y_train))
print('Train score : ',model.score(X_train,y_train))
print('Test score : ',model.score(X_test,y_test))
model = KNeighborsClassifier(n_neighbors = 3)
print(model.fit(X_train,y_train))
print('Train score : ',model.score(X_train,y_train))
print('Test score : ',model.score(X_test,y_test))
model = KNeighborsClassifier(n_neighbors = 6)
print(model.fit(X_train,y_train))
print('Train score : ',model.score(X_train,y_train))
print('Test score : ',model.score(X_test,y_test))

#Validation Set(Améliorer le model)permet de chercher les réglages du model qui donne les meilleurs  performances tout en gardant de cotés les données du test pour évuler le ML du test  

Val = cross_val_score(KNeighborsClassifier(),X_train,y_train, cv = 5,scoring = 'accuracy')
print(Val)
print(Val.mean())
Val = cross_val_score(KNeighborsClassifier(2),X_train,y_train, cv = 5,scoring = 'accuracy')
print(Val)
print(Val.mean())
Val = cross_val_score(KNeighborsClassifier(4),X_train,y_train, cv = 5,scoring = 'accuracy')
print(Val)
print(Val.mean())
Val_score =[]
for k in range(1,50):
    score = cross_val_score(KNeighborsClassifier(k),X_train,y_train, cv = 5,scoring = 'accuracy').mean()
    Val_score.append(score)
plt.plot(Val_score)
plt.show()

#validation_curve
model = KNeighborsClassifier()
k = np.arange(1, 50)
train_score, val_score = validation_curve(model,X_train, y_train,param_name= 'n_neighbors',param_range=k,cv=5)
print(train_score.shape)
print(val_score.shape)
print(val_score.mean(axis = 1))
plt.plot(k,val_score.mean(axis = 1), label ='Validation')
plt.plot(k,train_score.mean(axis = 1), label = 'Train')
plt.ylabel('score')
plt.xlabel('n_neighbors')
plt.legend()
plt.show()
# OverFitting le modele s'est trop perfectionner sur le trainset et a perdu tout sens de généralisation
### GridSearchCV construit une grille de modeles avec toutes les combinaisons d'hyperparametres presents dans param_frid
##
##param_grid = {'n_neighbors':np.arange(1,20),'motric':['eulidean','manhattan']}
##grid = GridSearchCV(KNeighborsClassifier(),param_grid, cv =5)
##print(grid)
##print(grid.fit(X_train, y_train))
##print(grid.score(X_test,y_test))


# Confusion Matrix outil de mesure trés utile pour évaluer la qualité d'un modele et montre les erreurs de classement

##print(confusion_matrix(y_test,model.predict(X_test)))

