import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

dataframe=pd.read_excel("embeddingsdata.xlsx")

binary_dataframe = dataframe[dataframe['Label'].isin([0, 1])]

X = binary_dataframe[['embed_1', 'embed_2']]
y = binary_dataframe['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

neigh = KNeighborsClassifier(n_neighbors=3)

neigh.fit(X_train, y_train)


accuracy = neigh.score(X_test, y_test)
print("Accuracy:", accuracy)


test_vector = X_test.iloc[0]  

predicted_class = neigh.predict([test_vector])

print("Predicted Class:", predicted_class[0])


