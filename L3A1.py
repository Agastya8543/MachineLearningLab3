import pandas as pd
import numpy as np

df = pd.read_excel('embeddingdata.xlsx')

class_a_data = df[df['Label'] == 0]
class_b_data = df[df['Label'] == 1]
intra_class_var_a = np.var(class_a_data[['embed_1', 'embed_2']], ddof=1)
intra_class_var_b = np.var(class_b_data[['embed_1', 'embed_2']], ddof=1)
mean_class_a = np.mean(class_a_data[['embed_1', 'embed_2']], axis=0)
mean_class_b = np.mean(class_b_data[['embed_1', 'embed_2']], axis=0)
inter_class_distance = np.linalg.norm(mean_class_a - mean_class_b)
print(f'Intraclass spread (variance) for Class A: {intra_class_var_a}')
print(f'Intraclass spread (variance) for Class B: {intra_class_var_b}')
print(f'Interclass distance between Class A and Class B: {inter_class_distance}')


# Print the results
print("Mean vector for Class A:")
print(mean_class_a)
print("\nStandard deviation for Class A:")
print(std_class_a)
print("\nMean vector for Class B:")
print(mean_class_b)
print("\nStandard deviation for Class B:")
print(std_class_b)
print("\nDistance between centroids of Class A and Class B:", distance_between_centroids)
