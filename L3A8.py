import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

dataframe=pd.read_excel("embeddingsdata.xlsx")
binary_dataframe = dataframe[dataframe['Label'].isin([0, 1])]

X = binary_dataframe[['embed_1', 'embed_2']]
y = binary_dataframe['Label']


accuracies_kNN = []
accuracies_NN = []

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
k_values = range(1, 12)

for k in k_values:
    
    kNN_classifier = KNeighborsClassifier(n_neighbors=k)
    kNN_classifier.fit(X_train, y_train)
    
    y_pred_kNN = kNN_classifier.predict(X_test)
       
    accuracy_kNN = accuracy_score(y_test, y_pred_kNN)
    accuracies_kNN.append(accuracy_kNN)
           
    NN_classifier = KNeighborsClassifier(n_neighbors=1)
    NN_classifier.fit(X_train, y_train)
               
    y_pred_NN = NN_classifier.predict(X_test)
                
    accuracy_NN = accuracy_score(y_test, y_pred_NN)
    accuracies_NN.append(accuracy_NN)

plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracies_kNN, marker='o', label='kNN (k=3)')
plt.plot(k_values, accuracies_NN, marker='o', label='NN (k=1)')

plt.title('Accuracy vs. k Value')
plt.xlabel('k Value')
plt.ylabel('Accuracy')
plt.xticks(k_values)
plt.legend()
plt.grid(True)
plt.show()
