import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report

dataframe=pd.read_excel("embeddingsdata.xlsx")
binary_dataframe = dataframe[dataframe['Label'].isin([0, 1])]

X = binary_dataframe[['embed_1', 'embed_2']]
y = binary_dataframe['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)
y_train_pred = neigh.predict(X_train)
confusion_matrix_train = confusion_matrix(y_train, y_train_pred)


y_test_pred = neigh.predict(X_test)
confusion_matrix_test = confusion_matrix(y_test, y_test_pred)


print("Confusion Matrix (Training Data):\n", confusion_matrix_train)
print("\nConfusion Matrix (Test Data):\n", confusion_matrix_test)


classification_report_train = classification_report(y_train, y_train_pred)
print("\nClassification Report (Training Data):\n", classification_report_train)


classification_report_test = classification_report(y_test, y_test_pred)
print("\nClassification Report (Test Data):\n", classification_report_test)