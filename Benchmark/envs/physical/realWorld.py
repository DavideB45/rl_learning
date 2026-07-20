import cv2
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random
import math
import time
import sys
import os
sys.path.append(os.path.join(sys.path[0], '..'))
#from control.safeControlBox import SafeControlBox
from envs.physical.control.mockControlBox import MockControlBox as SafeControlBox
from envs.physical.sense.pressureSensor import PressureSensor

class RealWorld(gym.Env):
	"""
	This is an environment that can be used to interact with the rel world
	"""


	def __init__(self, render_mode='rgb_array', width = 640, height = 480, target_size=80, approx_Hz=10, debug=False):
		'''
		initialize the environment by doing important initialization stuff (in the real world)
		'''
		super(RealWorld, self).__init__()
		self.width = width
		self.height = height
		self.target_size = target_size
		self.render_mode = render_mode
		self.debug = debug
		self.max_pressure = 1.4
		self.stepTime = 1/approx_Hz

		# Video stuff
		self.cap = cv2.VideoCapture(0)
		self.cap.set(3, self.width)
		self.cap.set(4, self.height)
		self.cap.set(10,150)
		arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
		arucoParams = cv2.aruco.DetectorParameters()
		self.arucoDetector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)

		# control box stuff
		self.box = SafeControlBox(max_pressure=self.max_pressure)
		if(not self.box.connect()):
			raise RuntimeError("Unable to connect to the controlbox, check the stuff and try again")
		
		# pressure sensor stuff
		self.pressure = PressureSensor()
		if(not self.pressure.connect()):
			raise RuntimeError("Unable to connect to the pressure sensor")

		# observation and action stuff
		self.action_space = spaces.Box( low=0, high=self.max_pressure, shape=(3,), dtype=np.float32 )
		self.observation_space = spaces.Box( low=0, high=self.max_pressure, shape=(4,), dtype=np.float32 )
		self.current_render = None
		self.current_prop = None
		self.current_step = 0

	def get_image(self, trials=30):
		success, img = self.cap.read()
		for _ in range(trials):
			if success:
				return img
			else:
				success = self.cap.read()
		return None
	
	def overlay_target(self, img):
		half_size = self.target_size // 2
		top_left_sq = (self.target_x - half_size, self.target_y - half_size)
		bottom_right_sq = (self.target_x + half_size, self.target_y + half_size)
		cv2.rectangle(img, top_left_sq, bottom_right_sq, (255, 0, 0), 2)
		cv2.circle(img, (self.target_x, self.target_y), 20, (255, 0, 0), -1)
		return img
	
	def get_arocu_rew(self, img) -> tuple[cv2.typing.MatLike, float]:
		corners, ids, _ = self.arucoDetector.detectMarkers(img)
		if ids is None or len(ids) > 1:
			raise RuntimeError("More than one ArUco detected")
		
		markerCorner, markerID = corners[0], ids[0]
		(topLeft, topRight, bottomRight, bottomLeft) =  markerCorner.reshape((4, 2))

		# 1. Calculate the center of the ArUco marker and draw a circle
		aruco_cX = int((topLeft[0] + bottomRight[0]) / 2.0)
		aruco_cY = int((topLeft[1] + bottomRight[1]) / 2.0)
		cv2.circle(img, (aruco_cX, aruco_cY), 4, (0, 255, 0), -1) 
		distance = math.hypot(self.target_x - aruco_cX, self.target_y - aruco_cY)
		rew = (self.height/2 - distance)/(self.height/2)
		
		if self.debug:
			# additional info about reward
			cv2.line(img, (aruco_cX, aruco_cY), (self.target_x, self.target_y), (0, 255, 255), 1)
			cv2.putText(img, f"Rew: {rew:.2f}", (aruco_cX - 35, aruco_cY - 15), 
						cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
		return img, rew

		
	def get_good_img(self) -> tuple[cv2.typing.MatLike, float]:
		done = False
		while not done:
			img = self.get_image(trials=30)
			try:
				img, rew = self.get_arocu_rew(img)
				self.overlay_target(img)
				done = True
			except RuntimeError:
				if self.debug:
					print("no ArUco found in current image, retrying")
		return img, rew

	
	def reset(self, seed=None, options=None):
		'''
		Reset the environment with a random target
		seed: random seed (actually ignored)
		options: additional options (really not used)
		returns: initial observation (np.array) still don't know what
		'''
		super().reset(seed=seed, options=options)
		self.box.reset()

		half_size = self.target_size // 2
		self.target_x = random.randint(half_size, self.width - half_size)
		self.target_y = random.randint(half_size, self.height - half_size)
		
		self.current_img, _ = self.get_good_img()
		self.current_prop = np.array([self.pressure.safe_read(), 0, 0, 0])
		self.current_step = 0
		return self.current_prop, {}

	def step(self, action) -> tuple:
		'''
		Step in the REAL world
		action: action to take
		returns: observation (np.array), reward (float), terminated (bool), truncated (bool), info (dict)
		'''
		# do the action
		self.box.send_pressure_array(action)
		time.sleep(self.stepTime)
		self.current_prop = np.array([self.pressure.safe_read(), action[0], action[1], action[2]])
		self.current_img, reward = self.get_good_img()
		info = {}
		if reward > 0.9:
			info['success'] = 1
		else:
			info['success'] = 0
		self.current_step += 1
		if(self.current_step > 200):
			terminated = True
		else:
			terminated = False
		return (
			self.current_prop,
			reward,
			terminated,
			False, # Terminated
			info
		)
	
	def render(self):
		if self.render_mode == 'rgb_array':
			return self.current_img
		elif self.render_mode == 'human':
			print('human')
			cv2.imshow("Result", self.current_img)
			cv2.waitKey(1)
			return self.current_img
		
	def close(self):
		self.cap.release()
		cv2.destroyAllWindows()
		self.box.reset()


if __name__ == "__main__":
	env = RealWorld(debug=True, render_mode='human')
	observation, _ = env.reset()
	total_reward = 0
	done = False
	while not done:
		action = env.action_space.sample()
		observation, reward, terminated, truncated, info = env.step(action)
		print(f"act: {action} rew:{reward}, obs:{observation}")
		env.render()
		if(info['success'] == 1):
			print(f'Win!! Total Reward: {total_reward}')
		done = terminated or truncated
		total_reward += reward
		if done:
			print(f"Game over! Total Reward: {total_reward}")
	env.close()