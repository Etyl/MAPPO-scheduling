import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from math import sqrt

data = []

for i in range(0,5):
    f = open(f'data/load_{i}.txt', "r")
    data.append([])
    for line in f:
        data[-1].append(float(line))
    f.close()


avg = []
var = []
for dataset in data:
    avg.append(sum(dataset)/len(dataset))
    var.append(sqrt(stats.describe(dataset).variance))

# Create scatter plot with error bars
plt.errorbar(range(0,5), avg, yerr=var, fmt='o', color='blue', capsize=5, markersize=8)
plt.xlabel('CPUs loaded at 100%')
plt.ylabel('Consumption (W)')

plt.show()