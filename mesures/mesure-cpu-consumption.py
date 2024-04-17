import paramiko
import time
import csv
import sys
from yoctopuce.yocto_api import *
from yoctopuce.yocto_power import *


saved_file_perf = "./data/data"
saved_file_consumption = "./data/consumption"
cores  = [1,2,3]
cpu_load = 100
ip_address = "192.168.1.153" # $ hostname -I
username = "admin"
password = "admin"
eventsLogged = ["branches", "instructions", "branch-misses"]
totalTime = 60 # s
interval = 100 # ms


taskset_cores = ",".join([str(core) for core in cores])

errmsg = YRefParam()

if YAPI.RegisterHub("127.0.0.1", errmsg) != YAPI.SUCCESS:
    sys.exit("init error: " + errmsg.value)

power = YPower.FirstPower()

if power is None : sys.exit("Can't detect the Yoctometter, please check it is plugged properly")

consumptions = []

ssh_client = paramiko.SSHClient()
ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh_client.connect(hostname=ip_address,username=username,password=password)
print("Successfully connected to", ip_address)

for cpu in cores:
    ssh_client.exec_command(f"stress-ng \
                            --cpu 1 \
                            --taskset {cpu} \
                            --timeout {totalTime+int(0.1*totalTime)}s \
                            --cpu-load {cpu_load}")
time.sleep(1)
ssh_client.exec_command(f"date +%s%N | cut -b1-13 > start-time\
                                                & sudo perf stat \
                                                    -o output \
                                                    --cpu {taskset_cores} \
                                                    -I {interval} \
                                                    --interval-count {totalTime*1000//interval} \
                                                    -e {','.join(eventsLogged)}")


start_time = time.time()
while time.time() - start_time < totalTime:
    consumptions.append(str(power.get_currentValue()) + " " + str(time.time()) + "\n")
    time.sleep(interval/(2*1000))

print("Successfully logged data")

stdin, stdout, stderr = ssh_client.exec_command("cat output")

timeout = 20

print("Attempting to retrieve perf data")
endtime = time.time() + timeout
while not stdout.channel.eof_received:
    time.sleep(1)
    print("Waiting for perf data...")
    if time.time() > endtime:
        stdout.channel.close()
        print("Timed out")
        break
output_data = stdout.read().decode(encoding='windows-1252').split('\n')
stdout.channel.close()
print("Successfully retrieved perf data")

print("Attempting to retrieve start time")
stdin, stdout, stderr = ssh_client.exec_command("cat start-time")
endtime = time.time() + timeout
while not stdout.channel.eof_received:
    time.sleep(1)
    print("Waiting for time data...")
    if time.time() > endtime:
        stdout.channel.close()
        print("Timed out")
        break
output_time = stdout.read().decode(encoding='windows-1252').strip()
stdout.channel.close()
startTime = float(output_time)/1000
print("Successfully retrieved start time")

ssh_client.close()

file_consumption = open(f"{saved_file_consumption}-cpu-{taskset_cores}-load-{cpu_load}.txt", "w")
for conso in consumptions:
    file_consumption.write(conso)
file_consumption.close()

# open the file in the write mode
file_perf = open(f"{saved_file_perf}-cpu-{taskset_cores}-load-{cpu_load}.csv", 'w', newline='')

# create the csv writer
writer = csv.writer(file_perf)

writer.writerow(['time', 'counts', 'events'])

for l in output_data[3:-1]:
    line = l.split()
    if line[0]=="#": continue
    row= []
    row.append(float(line[0])+startTime)
    k = 1
    n = ""
    while line[k].isdigit():
        n += line[k]
        k += 1
    row.append(int(n))
    row.append(line[k])
    if row[-1] in eventsLogged:
        writer.writerow(row)

# close the file
file_perf.close()
