import matplotlib.pyplot as plt
import numpy as np


bandwidths = [0,10000,15000,20000,25000,30000,40000]
consumption_avg = []
consumption_var = []

for b in bandwidths:
    loc = "./data/consumption-wifi-" + str(b) + ".txt"
    f = open(loc, "r")
    consumption = []
    for row in f:
        line = list(map(float,row.strip().split()))
        consumption.append(line[0])
    f.close()

    consumption_avg.append(sum(consumption)/len(consumption))
    consumption_var.append(np.var(consumption))

f, ax = plt.subplots(1)
plt.errorbar(bandwidths, consumption_avg , yerr=consumption_var, fmt='o', capsize=5, markersize=8)
plt.grid()
ax.set_ylim(ymin=0)
plt.xlabel("Bandwidth (kbit/s)")
plt.ylabel("Consumption (W)")

plt.savefig("./graphs/consumption-wifi-download.png")
#plt.show()
