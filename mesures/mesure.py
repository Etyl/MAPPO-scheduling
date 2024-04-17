import sys
from yoctopuce.yocto_api import *
import math
from yoctopuce.yocto_power import *
from time import *
import statistics as stats

# demarrer virtualhub control

filename = "data/load_3.txt"
interval = 0.5

errmsg = YRefParam()

if YAPI.RegisterHub("127.0.0.1", errmsg) != YAPI.SUCCESS:
    sys.exit("init error: " + errmsg.value)

power = YPower.FirstPower()

if power is None : sys.exit("Can't detect the Yoctometter, please check it is plugged properly")

f = open(filename, "w")
while True:
    f.write(str(power.get_currentValue()) + "\n")
    sleep(interval)