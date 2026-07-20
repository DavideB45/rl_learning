import serial
import time

class PressureSensor:
	def __init__(self, port='/dev/cu.usbmodem3101', baudrate=115200):
		"""Initializes the reader with default port settings."""
		self.port = port
		self.baudrate = baudrate
		self.ser = None
		self.min = 0
		self.max = 1023

	def connect(self):
		"""Attempts to connect to the Arduino. Returns True if successful."""
		try:
			print(f"Connecting to {self.port}...")
			self.ser = serial.Serial(self.port, self.baudrate, timeout=1)
			
			# Allow Arduino time to reboot
			time.sleep(2) 
			self.ser.reset_input_buffer()
			
			print("Connected successfully!")
			return True
		except serial.SerialException as e:
			print(f"Connection failed: {e}")
			self.ser = None
			return False
	
	def read(self):
		"""
		Requests a fresh reading from the Arduino and waits for the response.
		Highly efficient: zero buffer bloat.
		"""
		if not self.ser or not self.ser.is_open:
			print("Error: Port is not open.")
			return None
		try:
			self.ser.write(b'R') # 'send a request'
			line = self.ser.readline().decode('utf-8').strip() # read the fresh stuff
			if line:
				return (1 - int(line)/1023)
		except (ValueError, UnicodeDecodeError):
			return None
		except Exception as e:
			print(f"Error reading data: {e}")
			return None

		return None
	
	def safe_read(self):
		value = None
		while value is None:
			value = self.read()
		return value

	def disconnect(self):
		"""Safely closes the serial connection."""
		if self.ser and self.ser.is_open:
			self.ser.close()
			print("Serial port closed.")

# ==========================================
# Simple Usage Example
# ==========================================
if __name__ == "__main__":
	# Create the reader instance
	reader = PressureSensor()
	
	# Try connecting
	if reader.connect():
		print("Starting data log... (Press Ctrl+C to exit)\n")
		
		try:
			while True:
				value = reader.read()
				
				# Only print if a valid reading came through
				if value is not None:
					print(f"Sensor Value: {value}")
					
				# Small pause to keep your CPU running cool
				time.sleep(0.01)
				
		except KeyboardInterrupt:
			print("\nStopping...")
		finally:
			# Always clean up the connection at the end
			reader.disconnect()