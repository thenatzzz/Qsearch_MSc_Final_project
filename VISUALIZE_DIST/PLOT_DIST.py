import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
sns.set(color_codes=True)


data = pd.read_csv('original_model.csv')
data_accuary = data.iloc[:,-2].values
mean = np.mean(data_accuary)
std = np.std(data_accuary)
print("mean {} :: std {}".format(mean,std))

# bins = np.arange(0.5,0.85,0.005)
# hist, edges = np.histogram(data_accuary,bins)
# temp_dict = {}
# for i in range(len(hist)):
#     temp_dict[np.round(edges[i],6)] = hist[i]

# bins = np.arange(0.5,0.85,0.005)
bins = np.arange(mean,0.85,0.0004)

hist, edges = np.histogram(data_accuary,bins)
# print(hist)
# print(edges)

red_patch = mpatches.Patch(color='red', label='The red data')
mean_data = "Mean: {}".format(mean)
blue_patch = mpatches.Patch( label=mean_data)

plt.legend(handles=[red_patch,blue_patch])
# plt.plot([mean, mean], [0, 19], 'k-', lw=2)
plt.axvline(data_accuary.mean(), color='k', linestyle='dashed', linewidth=1)

sns.distplot(data_accuary)
plt.show()
