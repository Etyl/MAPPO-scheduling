import matplotlib.pyplot as plt

saved_file = "./data/data.csv"
eventsLogged = ["branches", "instructions", "branch-misses"]

T = []
events = dict()
for event in eventsLogged:
    events[event] = []

f = open(saved_file, "r")
f.readline()

for row in f:
    line = row.strip().split(",")
    if line[2] == eventsLogged[0]:
        T.append(float(line[0]))
    events[line[2]].append(int(line[1]))

for event in eventsLogged:
    plt.figure()
    plt.plot(T, events[event])
    plt.xlabel("Time (s)")
    plt.ylabel(event)
    plt.title(event)

plt.show()
