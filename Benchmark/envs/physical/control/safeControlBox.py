import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../../../'))
from time import sleep
from envs.physical.control.controlBox import ControlBox

class SafeControlBox(ControlBox):
	def __init__(self, max_pressure=0.5, slope=0.013, offset=-0.0103):
		"""
		Initializes the ControlBox object with no active connection.
		slope anf offset are the parameter of a line that fit the translation between 
		input and actual measured values. The default values are measured with the compressor at 3.5 bar
		"""
		super().__init__(slope=slope, offset=offset)
		self.max_pressure = max_pressure


	def send_raw(self, d1, d2, d3):
		"""
		Sends raw integer values (0-255) to the Arduino. 
		Useful for calibration and finding formula parameters.
		"""
		raise NotImplementedError("send raw is not available when using the SafeControlBox")

	def send_pressure(self, v1, v2, v3):
		"""
		Takes desired pressure as floats, applies the calibration math, 
		and sends the command.
		"""
		d1 = self.bar_to_raw(min(self.max_pressure, v1))
		d2 = self.bar_to_raw(min(self.max_pressure, v2))
		d3 = self.bar_to_raw(min(self.max_pressure, v3))
		
		# Pass the calculated values to the raw sender
		super().send_raw(d1, d2, d3)

	def reset(self):
		self.send_pressure(0, 0, 0)

	def disconnect(self):
		self.reset()

	def send_pressure_array(self, pressure):
		self.send_pressure(pressure[0], pressure[1], pressure[2])

if __name__ == "__main__":
	# Create an instance of our control box
	max_pressure = 0.8
	chamber_idx = 1
	tot = 10
	box = SafeControlBox(max_pressure=max_pressure)
	box.connect()
	for i in range(tot+1):
		box.send_pressure(
			i/tot*max_pressure if chamber_idx == 0 else 0.0, 
			i/tot*max_pressure if chamber_idx == 1 else 0.0, 
			i/tot*max_pressure if chamber_idx == 2 else 0.0)
		print(f"Sent pressure: \033[34m{i/tot*max_pressure:.3f}\033[0m bar to chamber {chamber_idx}")
		sleep(0.5)
	box.send_pressure(0, 0, 0)