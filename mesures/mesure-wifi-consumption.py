import paramiko
import time
import sys
from yoctopuce.yocto_api import *
from yoctopuce.yocto_power import *
import os

## launch iperf -s on the target machine

saved_file_consumption = "./data/consumption"
ip_address = "192.168.143.190" # $ hostname -I
username = "admin"
password = "admin"
totalTime = 60 # s
bandwidth = 15000 # kbit/s
interval = 100 # ms


errmsg = YRefParam()

if YAPI.RegisterHub("127.0.0.1", errmsg) != YAPI.SUCCESS:
    sys.exit("init error: " + errmsg.value)

power = YPower.FirstPower()

if power is None : sys.exit("Can't detect the Yoctometter, please check it is plugged properly")

consumptions = []

start_time = time.time()
while time.time() - start_time < totalTime:
    consumptions.append(str(power.get_currentValue()) + " " + str(time.time()) + "\n")
    time.sleep(interval/(2*1000))

print("Successfully logged data")


file_consumption = open(f"{saved_file_consumption}-wifi-{bandwidth}.txt", "w")
for conso in consumptions:
    file_consumption.write(conso)
file_consumption.close()

