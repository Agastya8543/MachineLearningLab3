import numpy as np
import pandas as pd
df=pd.read_excel("embeddingsdata.xlsx")

class_x_data = df[df['Label'] == 0]  
class_y_data = df[df['Label'] == 1]  
intra_class_var_x = np.var(class_x_data[['embed_1', 'embed_2']], ddof=1)  
intra_class_var_y = np.var(class_y_data[['embed_1', 'embed_2']], ddof=1)  
mean_class_x = np.mean(class_x_data[['embed_1', 'embed_2']], axis=0)  
mean_class_y = np.mean(class_y_data[['embed_1', 'embed_2']], axis=0)  
inter_class_distance = np.linalg.norm(mean_class_x - mean_class_y)
print(f'Intraclass spread (variance) for Class X: {intra_class_var_x}')
print(f'Intraclass spread (variance) for Class Y: {intra_class_var_y}')
print(f'Interclass distance between Class X and Class Y: {inter_class_distance}')




unique_classes = df['Label'].unique()
class_centroids = {}

for class_label in unique_classes:
    class_data = df[df['Label'] == class_label]
    class_mean = np.mean(class_data[['embed_1', 'embed_2']], axis=0)
    class_centroids[class_label] = class_mean

for class_label, centroid in class_centroids.items():
    print(f'Class {class_label} Centroid: {centroid}')





grouped = df.groupby('Label')
class_standard_deviations = {}
for class_label, group_data in grouped:
    class_std = group_data[['embed_1', 'embed_2']].std(axis=0)
    class_standard_deviations[class_label] = class_std
for class_label, std_deviation in class_standard_deviations.items():
    print(f'Standard Deviation for Class {class_label}:')
    for col, std in zip(std_deviation.index, std_deviation.values):
        print(f'  {col}: {std}')


# In[8]:


grouped = df.groupby('Label')


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