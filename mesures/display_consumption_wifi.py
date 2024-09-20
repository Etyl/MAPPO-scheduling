import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats


bandwidths_eth = [0,10000,20000,30000,40000,50000,70000,80000,90000]
bandwidths_wifi = [0, 10000, 15000, 20000, 25000, 30000, 40000]
consumption_avg = []
consumption_var = []

for b in bandwidths_eth:
    loc = "./data/consumption-eth-" + str(b) + ".txt"
    f = open(loc, "r")
    consumption = []
    for row in f:
        line = list(map(float,row.strip().split()))
        consumption.append(line[0])
    f.close()

    consumption_avg.append(sum(consumption)/len(consumption))
    consumption_var.append(np.var(consumption))


lr = stats.linregress(bandwidths_eth, consumption_avg)
avg = np.array(bandwidths_eth)
f, ax = plt.subplots(1)
plt.errorbar(bandwidths_eth, consumption_avg , yerr=consumption_var, fmt='o', capsize=5, markersize=8)
#plt.plot(avg, lr.intercept + lr.slope*avg, 'r', label='fitted line')
plt.grid()
ax.set_ylim(ymin=0,ymax=4.2)
ax.set_xlim(xmin=0,xmax=90000)
#plt.legend()
plt.xlabel("Bandwidth (kbit/s)")
plt.ylabel("Consumption (W)")
plt.savefig("./graphs/consumption-eth-download.svg", format="svg")
