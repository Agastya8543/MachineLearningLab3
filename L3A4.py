import pandas as pd
from sklearn.model_selection import train_test_split

dataframe=pd.read_excel("embeddingsdata.xlsx")
binary_dataframe=dataframe[dataframe['Label'].isin([0, 1])]

X = binary_dataframe[['embed_1', 'embed_2']]
y = binary_dataframe['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

