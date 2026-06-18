if __name__ == '__main__':
	import time
	tot = 0
	times = {
	"collecting_time": 28281.488332509995,
	"vq_training_time": 12017.390539646149,
	"lstm_training_time": 8958.13176369667,
	"dataset_generation_time": 16943.998064756393,
	"agent_training_time": 6480.0694744586945,
	"evaluation_time": 18568.107147216797
	}
	times = {
	"collecting_time": 28852.194244623184,
	"vq_training_time": 5033.064501047134,
	"lstm_training_time": 4761.03252863884,
	"dataset_generation_time": 8425.21964931488,
	"agent_training_time": 7343.691750526428
	}
	times = {
 "collecting_time": 23854.321676015854,
 "vq_training_time": 9808.06747674942,
 "lstm_training_time": 22046.566100358963,
 "dataset_generation_time": 19469.181207180023,
 "agent_training_time": 45044.86441373825
}
	for key in times:
		tot += times[key]
		days = int(times[key] // 86400)
		print(f"{key} : {days}d {time.strftime('%H:%M:%S', time.gmtime(times[key]))}")
	days = int(tot // 86400)
	print(f"tot : {days}d {time.strftime('%H:%M:%S', time.gmtime(tot))}")