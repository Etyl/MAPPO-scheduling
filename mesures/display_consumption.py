import matplotlib.pyplot as plt
import numpy as np

interval = 0.5

data = []

for i in range(0,5):
    f = open(f'data/load_{i}.txt', "r")
    data.append([])
    for line in f:
        data[-1].append(float(line))
    f.close()

L = np.Infinity
for dataset in data:
    L = min(L, len(dataset))

T = np.linspace(0, L*interval, L)

for i in range(0,5):
    plt.plot(T, data[i][:L], label=f'{i} CPU loaded at 100%')
plt.xlabel('Time (s)')
plt.ylabel('Consumption (W)')
plt.legend()
plt.show()