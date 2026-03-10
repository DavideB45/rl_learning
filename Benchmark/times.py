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
	for key in times:
		tot += times[key]
		print(f"{key} : {time.strftime('%H:%M:%S', time.gmtime(times[key]))}")
	print(f"tot : {time.strftime('%H:%M:%S', time.gmtime(tot))}")