import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataframe=pd.read_excel("embeddingsdata.xlsx")
feature1_data = dataframe['embed_1']

num_bins = 5
hist_counts, bin_edges = np.histogram(feature1_data, bins=num_bins)
mean_feature1 = np.mean(feature1_data)
variance_feature1 = np.var(feature1_data, ddof=1)

plt.hist(feature1_data, bins=num_bins, edgecolor='black', alpha=0.7)
plt.xlabel('Feature1')
plt.ylabel('Frequency')
plt.title('Histogram of Feature1')
plt.grid(True)
plt.show()
print(f'Mean of Feature1: {mean_feature1}')
print(f'Variance of Feature1: {variance_feature1}')