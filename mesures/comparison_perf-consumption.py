import matplotlib.pyplot as plt

cpu_load = 50
cores = [1,2]
saved_file_perf = "./data/data"
saved_file_consumption = "./data/consumption"
eventsLogged = ["branches", "instructions", "branch-misses"]

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


fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Temps (s)')
ax1.set_ylabel('Consommation (W)', color=color)
ax1.plot(consumption_time, consumption, color=color)
ax1.tick_params(axis='y', labelcolor=color)


ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Instructions', color=color)
ax2.plot(events["time"], events["instructions"], color=color)
ax2.tick_params(axis='y', labelcolor=color)

plt.show()

"""
for event in eventsLogged:
    plt.figure()
    plt.plot(events[event], consumption, ".")
    plt.xlabel(event)
    plt.ylabel("Consumption (W)")
"""

plt.show()

