import matplotlib.pyplot as plt
from math import sqrt
import scipy.stats as stats
import numpy as np

cpu_loads = [0,20,50,80,100]
cores_list = [[1],[1,2],[1,2,3]]
saved_file_perf = "./data/data"
saved_file_consumption = "./data/consumption"
eventsLogged = ["branches", "instructions", "branch-misses"]

xpos = []
ypos = []
zpos = []

fig,ax = plt.subplots(1,1)

consumption_avg= []
consumption_var = []
events_avg = dict()
for event in eventsLogged:
    events_avg[event] = []

for cores in cores_list:
    for cpu_load in cpu_loads:
        xpos.append(len(cores)-0.25)
        ypos.append(cpu_load-5)
        print(f"Plotting cpu_load={cpu_load} and cores={cores}")
        taskset_cores = ",".join([str(core) for core in cores])

        events = dict()
        for event in eventsLogged:
            events[event] = []
        events["time"] = []

        f = open(f"{saved_file_perf}-cpu-{taskset_cores}-load-{cpu_load}.csv", "r")
        f.readline()

        for row in f:
            line = row.strip().split(",")
            if line[2] == eventsLogged[0]:
                events["time"].append(float(line[0]))
            events[line[2]].append(int(line[1]))

        f.close()

        f = open(f"{saved_file_consumption}-cpu-{taskset_cores}-load-{cpu_load}.txt", "r")
        consumption = []
        consumption_time = []
        for row in f:
            line = list(map(float,row.strip().split()))
            consumption.append(line[0])
            consumption_time.append(line[1])
        f.close()


        startTime = min(events["time"][0], consumption_time[0])

        for i in range(len(events["time"])):
            events["time"][i] -= startTime
        for i in range(len(consumption)):
            consumption_time[i] -= startTime

        avg = sum(consumption)/len(consumption)
        zpos.append(avg)

        consumption_avg.append(avg)
        consumption_var.append(sqrt(stats.describe(consumption).variance))
        for event in eventsLogged:
            events_avg[event].append(sum(events[event])/len(events[event]))

print(events_avg["branch-misses"])

# Create scatter plot with error bars
for event in eventsLogged:
    avg = events_avg[event]
    f, ax = plt.subplots(1)
    plt.errorbar(events_avg[event], consumption_avg , yerr=consumption_var, fmt='o', capsize=5, markersize=8, label=event)
    plt.grid()
    ax.set_ylim(ymin=0,ymax=4.2)
    plt.xlabel(event)
    plt.ylabel('Consumption (W)')
    plt.savefig(f"./graphs/perf-consumption-{event}.png")


lr = stats.linregress(events_avg["instructions"], consumption_avg)
avg = np.array(events_avg["instructions"])
f, ax = plt.subplots(1)
plt.errorbar(events_avg["instructions"], consumption_avg , yerr=consumption_var, fmt='o', capsize=5, markersize=8)
plt.plot(avg, lr.intercept + lr.slope*avg, 'r', label='fitted line')
plt.grid()
ax.set_ylim(ymin=0,ymax=4.2)
plt.legend()
plt.xlabel("instructions")
plt.ylabel('Consumption (W)')
plt.savefig(f"./graphs/perf-consumption-instructions-linregress.svg", format="svg")



fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# Construct arrays with the dimensions for the 16 bars.
dx = 0.5 * np.ones_like(zpos)
dy = 10 * np.ones_like(zpos)
dz = zpos

ax.bar3d(xpos, ypos, 0, dx, dy, dz, zsort='average')

plt.savefig("./graphs/perf-consumption-3d.png")

