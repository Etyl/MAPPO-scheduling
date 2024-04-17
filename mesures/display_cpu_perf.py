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
    f, ax = plt.subplots(1)
    ax.plot(T, events[event])
    plt.grid()
    plt.xlabel("Time (s)")
    plt.ylabel(event)
    ax.set_ylim(bottom=0)
    plt.title(event)

plt.show()
