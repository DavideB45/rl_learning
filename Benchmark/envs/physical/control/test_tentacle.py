import time
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.join(sys.path[0], '..'))
from control.safeControlBox import SafeControlBox

MAX_PRESSURE = 1.5
box = SafeControlBox(max_pressure=MAX_PRESSURE)
if(not box.connect()):
	raise RuntimeError("Unable to connect to the controlbox, check the stuff and try again")
try:
	for i in tqdm(range(0, 101), colour='blue'):
		time.sleep(0.1)
		pressure = MAX_PRESSURE*i/100
		box.send_pressure(0, pressure, pressure) # last one is mesuring
except KeyboardInterrupt:
	box.reset()
time.sleep(1)
box.reset()



