import os
import pandas as pd
from datetime import datetime
import shutil



to_split = '2025-03-12_13.21.00_Hum_2_1_x_SD_Session1_TO_FIX/Hum_2_1_x_Session1_embeds13_Calibrated_SD.csv'

to_rename = [
	'2025-03-12_13.11.39_Hum_2_1_x2_SD_Session1',
	'2025-03-12_13.18.44_Hum_2_1_x1_SD_Session1',
	'2025-03-12_13.21.00_Hum_2_1_x_SD_Session1', # -> This do not exist at the begining of the script
	'2025-03-12_14.16.00_Hum_2_2_x_SD_Session1', # -----^
	'2025-03-12_14.14.35_Hum_2_2_x1_SD_Session1',
	'2025-03-12_14.20.39_Hum_2_2_x2_SD_Session1'
]

print("\033[91mWARNING: you can not reverse this process, be sure to have duplicates of your data\033[0m")
print("Do you want to continue? [Y/n] ")
answer = input().strip().lower()
if answer not in ['y', 'yes']:
	print("Aborted.")
	exit(0)

# first line of the csv tells us the delimiter, the third line tells us the unit of measure
df = pd.read_csv(to_split, skiprows=[0,2], sep='\t')

# Convert the first timestamp to human-readable format 
# it is in ms not seconds, so we need to divide by 1000
timestamp = df['embeds13_Timestamp_Unix_CAL'].iloc[0]
human_readable = datetime.fromtimestamp(timestamp / 1000).strftime('%Y-%m-%d %H:%M:%S')
print(f"First timestamp (human-readable): {human_readable}")

print('Splitting the file in 2 different sessions with the second one starting at 14:16:00')
split_time_str = '2025-03-12 14:16:00'
split_timestamp = int(datetime.strptime(split_time_str, '%Y-%m-%d %H:%M:%S').timestamp() * 1000)
print(f"Split timestamp (ms): {split_timestamp}")

# Divide the dataframe into two parts based on the split_timestamp
df_session1 = df[df['embeds13_Timestamp_Unix_CAL'] < split_timestamp]
df_session2 = df[df['embeds13_Timestamp_Unix_CAL'] >= split_timestamp]

#print(f"Session 1 rows: {len(df_session1)}")
#print(f"Session 2 rows: {len(df_session2)}")

# Save the first session to a new CSV file in the new folder
os.makedirs('2025-03-12_13.21.00_Hum_2_1_x_SD_Session1', exist_ok=True)
session1_file = '2025-03-12_13.21.00_Hum_2_1_x_SD_Session1/data_session1.csv'
df_session1.to_csv(session1_file, index=False)

# Save the second session to a new CSV file in the new folder
os.makedirs('2025-03-12_14.16.00_Hum_2_2_x_SD_Session1', exist_ok=True)
session2_file = '2025-03-12_14.16.00_Hum_2_2_x_SD_Session1/data_session2.csv'
df_session2.to_csv(session2_file, index=False)

correct_names = [
	'2025-03-12_13.11.39_gton3035_x2_SD_Session1',
	'2025-03-12_13.18.44_gton3035_x1_SD_Session1',
	'2025-03-12_13.21.00_gton3035_x_SD_Session1',

	'2025-03-12_14.16.00_0llo84vt_x_SD_Session1',
	'2025-03-12_14.14.35_0llo84vt_x1_SD_Session1',
	'2025-03-12_14.20.39_0llo84vt_x2_SD_Session1'
]

print("\033[1;34mRenaming folders to correct session IDs...\033[0m")
print("\033[93mThis is your last chance to abort before renaming folders. Do you want to continue? [Y/n] \033[0m")
final_answer = input().strip().lower()
if final_answer not in ['y', 'yes']:
	print("Aborted.")
	exit(0)
for old_name, new_name in zip(to_rename, correct_names):
	print(f"Renaming {old_name} to {new_name}")
	os.rename(old_name, new_name)
	# Delete the folder 2025-03-12_13.21.00_Hum_2_1_x_SD_Session1_TO_FIX if it exists
folder_to_delete = '2025-03-12_13.21.00_Hum_2_1_x_SD_Session1_TO_FIX'
if os.path.exists(folder_to_delete) and os.path.isdir(folder_to_delete):
	shutil.rmtree(folder_to_delete)
	print(f"Deleted folder: {folder_to_delete}")
else:
	print(f"Folder not found: {folder_to_delete}")
print("\033[1;32mDone!\033[0m")