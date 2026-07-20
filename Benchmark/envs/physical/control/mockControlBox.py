from control.controlBox import ControlBox

class MockControlBox(ControlBox):
	def __init__(self, max_pressure=0.5, slope=0.013, offset=-0.0103):
		"""
		Initializes the ControlBox object with no active connection.
		slope anf offset are the parameter of a line that fit the translation between 
		input and actual measured values. The default values are measured with the compressor at 3.5 bar
		"""
		self.slope = slope
		self.offset = offset

	def get_available_ports(self):
		"""Returns a list of available serial port names."""
		return []

	def connect(self, target_port='/dev/cu.usbmodem3101'):
		"""
		Tries to connect to the target_port automatically. 
		If it's not found, lists available ports and prompts the user.
		"""
		selected_port = None
		print(f"Found preferred port: {target_port}. Attempting connection...")
		print(f"Successfully connected to {selected_port}!")
		return True


	def send_raw(self, d1, d2, d3):
		"""
		Sends raw integer values (0-255) to the Arduino. 
		Useful for calibration and finding formula parameters.
		"""
		pass

	def raw_to_bar(self, bit):
		return bit*self.slope - self.offset

	def bar_to_raw(self, v):
		return round((v + self.offset) / self.slope)

	def send_pressure(self, v1, v2, v3):
		"""
		Takes desired pressure as floats, applies the calibration math, 
		and sends the command.
		"""
		pass

	def disconnect(self):
		"""Safely closes the connection to the control box."""
		print("Disconnected from Control Box.")

	def reset(self):
		pass

	def send_pressure_array(self, pressure):
		pass