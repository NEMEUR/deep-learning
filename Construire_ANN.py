# -*- coding: utf-8 -*-
"""
Created on Sun May 17 13:55:24 2020

@author: SABRINA NEMEUR
"""

# Importer les modules
import pandas as pd
import numpy as np

# Importer la data
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13]
y = dataset.iloc[:, 13]

# Encoder les données catégoriques et standarider les valeurs continues
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
preprocess = make_column_transformer(
        (OneHotEncoder(), ['Geography', 'Gender']),
        (StandardScaler(), ['CreditScore', 'Age', 'Tenure', 'Balance',
                            'NumOfProducts', 'HasCrCard', 'IsActiveMember', 
                            'EstimatedSalary']))


X = preprocess.fit_transform(X)


X = np.delete(X, [0,3], 1)

y = y.values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#  construire notre ANN!

# Importer les modules  
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Initialisation du ANN
classifier = Sequential()

# Ajouter la couche d'entrée et la première couche cachée
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
classifier.add(Dropout(rate=0.1))
# Ajouter la seconde couche cachée
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dropout(rate=0.1))
# Ajouter la couche de sortie
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiler notre ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Lancer l'entrainement de notre ANN sur le training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)


# prédire sur le test set 
y_pred = classifier.predict(X_test)

y_pred=(y_pred>0.5)

# mesurer la performance de notre ANN avec la matrix de confusion
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# faire une prédiction sur une nouvelle observation
"""prédire si un nouveau client va quitter la banque avec les données suivantes:
Geography: France
Credit Score:: 600
Gender: Male
Age: 40
Tenure: 3
Balance: 60000
Number of Products: 2
Has Credit Card: Yes
Is Active Member: Yes
Estimated Salary: 50000"""

Xnew = pd.DataFrame(data={
        'CreditScore': [600], 
        'Geography': ['France'], 
        'Gender': ['Male'],
        'Age': [40],
        'Tenure': [3],
        'Balance': [60000],
        'NumOfProducts': [2],
        'HasCrCard': [1],
        'IsActiveMember': [1],
        'EstimatedSalary': [50000]})
Xnew = preprocess.transform(Xnew)
Xnew = np.delete(Xnew, [0,3], 1)
new_prediction=classifier.predict(Xnew)
new_prediction = (new_prediction > 0.5)

#améliorer notre ANN avec le KFOLD cross validation
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score


def build_classifier(optimizer='adam'):
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

# Evaluer notre ANN

classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, 
                             cv = 10 )#n_jobs=-1 tous les cpu

mean = accuracies.mean()
variance = accuracies.std() 


#ajuster les hyperparamètres
from sklearn.model_selection import GridSearchCV

classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [25, 32],
              'epochs': [100, 500],
              'optimizer': ['adam', 'rmsprop']}
#pour chaque combinaison (8)il divise le train en kfolds=10
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)#folds=10
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
