import numpy as np
import pandas as pd

dataframe=pd.read_excel("embeddingsdata.xlsx")

class_a_data = dataframe[dataframe['Label'] == 0]  
class_b_data = dataframe[dataframe['Label'] == 1]  
intra_class_var_a = np.var(class_a_data[['embed_1', 'embed_2']], ddof=1)  
intra_class_var_b = np.var(class_b_data[['embed_1', 'embed_2']], ddof=1)  
mean_class_a = np.mean(class_a_data[['embed_1', 'embed_2']], axis=0)  
mean_class_b = np.mean(class_b_data[['embed_1', 'embed_2']], axis=0)  
inter_class_distance = np.linalg.norm(mean_class_a - mean_class_b)
print(f'Intraclass spread (variance) for Class A: {intra_class_var_a}')
print(f'Intraclass spread (variance) for Class B: {intra_class_var_b}')
print(f'Interclass distance between Class A and Class B: {inter_class_distance}')

unique_classes = dataframe['Label'].unique()
class_centroids = {}

for class_label in unique_classes:
    class_data = dataframe[dataframe['Label'] == class_label]
    class_mean = np.mean(class_data[['embed_1', 'embed_2']], axis=0)
    class_centroids[class_label] = class_mean

for class_label, centroid in class_centroids.items():
    print(f'Class {class_label} Centroid: {centroid}')

grouped = dataframe.groupby('Label')
class_standard_deviations = {}

for class_label, group_data in grouped:
    class_std = group_data[['embed_1', 'embed_2']].std(axis=0)
    class_standard_deviations[class_label] = class_std
for class_label, std_deviation in class_standard_deviations.items():
    print(f'Standard Deviation for Class {class_label}:')
    for col, std in zip(std_deviation.index, std_deviation.values):
        print(f'  {col}: {std}')

grouped = dataframe.groupby('Label')
class_centroids = {}

for class_label, group_data in grouped:
    class_mean = group_data[['embed_1', 'embed_2']].mean(axis=0)
    class_centroids[class_label] = class_mean

class_labels = list(class_centroids.keys())
num_classes = len(class_labels)
class_distances = {}

for i in range(num_classes):
    for j in range(i + 1, num_classes):
        class_label1 = class_labels[i]
        class_label2 = class_labels[j]
        distance = np.linalg.norm(class_centroids[class_label1] - class_centroids[class_label2])
        class_distances[(class_label1, class_label2)] = distance

for (class_label1, class_label2), distance in class_distances.items():
    print(f'Distance between Class {class_label1} and Class {class_label2}: {distance}')