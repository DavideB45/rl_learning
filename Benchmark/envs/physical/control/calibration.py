import csv
import os
import time
from controlBox import ControlBox

if __name__ == "__main__":
    # Create an instance of our control box
    box = ControlBox()
    box.connect()
    
    values = []
    file_name = 'calibration_readings.csv'

    # 1. INCREASING LOOP (0 to 255)
    print("\n=== Starting INCREASING Calibration ===")
    for i in range(0, 256, 15):  # 256 ensures it includes 255
        print(f"Sending {i} (approx {box.raw_to_bar(i)} bar) [Increasing]")
        box.send_raw(i, 0, 0)
        try:
            user_read = float(input("Enter the pressure reading (float): "))
        except Exception:
            print("Invalid input, using 0.0")
            user_read = 0.0
        values.append((i, user_read, 'increasing'))

    # 2. DECREASING LOOP (255 down to 0)
    print("\n=== Starting DECREASING Calibration ===")
    for i in range(255, -1, -15):  # -1 ensures it includes 0
        print(f"Sending {i} (approx {box.raw_to_bar(i)} bar) [Decreasing]")
        box.send_raw(i, 0, 0)
        try:
            user_read = float(input("Enter the pressure reading (float): "))
        except Exception:
            print("Invalid input, using 0.0")
            user_read = 0.0
        values.append((i, user_read, 'decreasing'))


	# Turn off the stuff
    box.send_raw(0, 0, 0)


    file_exists = os.path.exists(file_name)
    with open(file_name, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(['raw_value', 'user_reading', 'direction'])
        writer.writerows(values)
    print(f"\nSuccess! Data appended to '{file_name}'")
		
		