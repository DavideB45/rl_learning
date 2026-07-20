import serial
import serial.tools.list_ports
import time

class ControlBox:
	def __init__(self, slope=0.013, offset=-0.0103):
		"""
		Initializes the ControlBox object with no active connection.
		slope anf offset are the parameter of a line that fit the translation between 
		input and actual measured values. The default values are measured with the compressor at 3.5 bar
		"""
		self.dev = None
		self.slope = slope
		self.offset = offset

	def get_available_ports(self):
		"""Returns a list of available serial port names."""
		ports = serial.tools.list_ports.comports()
		return [port.device for port in ports]

	def connect(self, target_port='/dev/cu.usbmodem3101'):
		"""
		Tries to connect to the target_port automatically. 
		If it's not found, lists available ports and prompts the user.
		"""
		available_ports = self.get_available_ports()
		
		if not available_ports:
			print("No serial ports found! Check your USB connection.")
			return False

		selected_port = None

		# 1. Try to automatically connect to your preferred port
		if target_port in available_ports:
			print(f"Found preferred port: {target_port}. Attempting connection...")
			selected_port = target_port
		else:
			# 2. Fall back to user prompt if the preferred port isn't there
			print(f"Preferred port '{target_port}' not found.")
			print("These are the currently available ports:")
			for i, port in enumerate(available_ports):
				print(f"[{i}] {port}")
			
			try:
				selection = input("Which port index is the Control Box connected to? ")
				selected_port = available_ports[int(selection)]
			except (ValueError, IndexError):
				print("Invalid selection. Aborting connection.")
				return False

		# 3. Open the port
		try:
			self.dev = serial.Serial(
				port=selected_port,
				baudrate=115200,
				bytesize=serial.EIGHTBITS,
				stopbits=serial.STOPBITS_ONE,
				parity=serial.PARITY_NONE,
				timeout=1
			)
			print(f"Successfully connected to {selected_port}!")
			
			# Wait for Arduino to reset upon connection
			time.sleep(2) 
			return True
			
		except serial.SerialException as e:
			print(f"Error opening port {selected_port}: {e}")
			self.dev = None
			return False

	def send_raw(self, d1, d2, d3):
		"""
		Sends raw integer values (0-255) to the Arduino. 
		Useful for calibration and finding formula parameters.
		"""
		if not self.dev or not self.dev.is_open:
			print("Error: Cannot send data, Control Box is not connected.")
			return
		
		# Clamp values to 0-255 to prevent crashes or overflow
		d1 = max(0, min(255, int(d1)))
		d2 = max(0, min(255, int(d2)))
		d3 = max(0, min(255, int(d3)))
		
		data_packet = bytearray([106, d1, d2, d3])
		
		try:
			self.dev.write(data_packet)
		except Exception as e:
			print(f"Error writing to COM port: {e}")

	def raw_to_bar(self, bit):
		return bit*self.slope - self.offset

	def bar_to_raw(self, v):
		return round((v + self.offset) / self.slope)

	def send_pressure(self, v1, v2, v3):
		"""
		Takes desired pressure as floats, applies the calibration math, 
		and sends the command.
		"""
		d1 = self.bar_to_raw(v1)
		d2 = self.bar_to_raw(v2)
		d3 = self.bar_to_raw(v3)
		
		# Pass the calculated values to the raw sender
		self.send_raw(d1, d2, d3)

	def disconnect(self):
		"""Safely closes the connection to the control box."""
		if self.dev and self.dev.is_open:
			# Good practice: set valves to 0 before closing the connection
			print("Setting valves to 0 before disconnecting...")
			self.send_raw(0, 0, 0)
			time.sleep(0.1) # Brief pause to ensure the final command sends
			
			self.dev.close()
			print("Disconnected from Control Box.")


# ==========================================
# Example usage of the class
# ==========================================
if __name__ == "__main__":
	# Create an instance of our control box
	box = ControlBox()
	
	# Attempt to connect (will look for '/dev/cu.usbmodem3101' by default)
	if box.connect():
		
		# --- CALIBRATION MODE EXAMPLE ---
		# Send raw 8-bit integers directly to the hardware
		print("Sending raw calibration data...")
		box.send_raw(0, 0, 0)
		time.sleep(4)
		
		# Always close the connection when you're done!
		box.disconnect()