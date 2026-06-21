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
 "collecting_time": 9111.88158750534,
 "vq_training_time": 3857.5122294425964,
 "lstm_training_time": 11892.542912721634,
 "dataset_generation_time": 7641.496007680893,
 "agent_training_time": 24671.919432640076
}
	for key in times:
		tot += times[key]
		days = int(times[key] // 86400)
		print(f"{key} : {days}d {time.strftime('%H:%M:%S', time.gmtime(times[key]))}")
	days = int(tot // 86400)
	print(f"tot : {days}d {time.strftime('%H:%M:%S', time.gmtime(tot))}")