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

	def send_pressure_array(self, pressure):
		self.send_pressure(pressure[0], pressure[1], pressure[2])