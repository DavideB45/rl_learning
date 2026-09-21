import cv2
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from PIL import Image
import time
import sys
import os
sys.path.append(os.path.join(sys.path[0], '../..'))
sys.path.append(os.path.join(sys.path[0], '..'))
from envs.physical.control.safeControlBox import SafeControlBox
#from envs.physical.control.mockControlBox import MockControlBox as SafeControlBox
#from envs.physical.sense.pressureSensor import PressureSensor
from envs.physical.sense.ArucoRotation import ArucoRotationTracker
from envs.physical.sense.CroppingCamera import CroppingCamera

class RealWorld(gym.Env):
	"""
	This is an environment that can be used to interact with the rel world
	"""


	def __init__(self, 
			  view_camera_id=1, width = 640, height = 480, cropped_width=64, cropped_height=64, camera_hz=20,
			  aruco_camera_id=0, marker_id=None, min_sharpness=100.0, aruco_hz=20,
			  render_mode='rgb_array', max_steps=100, env_hz=10, debug=False, rew_multiplier=8.0):
		'''
		initialize the environment by doing important initialization stuff (in the real world)
		'''
		super(RealWorld, self).__init__()
		self.width = width
		self.height = height
		self.render_mode = render_mode
		self.reward_multiplier = rew_multiplier
		self.debug = debug
		self.max_pressure = 1.3
		if debug:
			self.max_pressure = 0.5
		self.stepTime = 1/env_hz
		self.max_steps = max_steps
		self.current_pressure = np.array([0, 0, 0])

		# Video Capture
		self.camera = CroppingCamera(camera_index=view_camera_id, width=self.width, height=self.height, resized_width=cropped_width, resized_height=cropped_height, rate_hz=camera_hz, debug=debug)
		self.camera.start(wait=True, timeout=5.0)
		self.current_img, self.cropped_img = self.camera.get_clear_image(timeout=5.0)
		# Reward Camera
		self.arucoDetector = ArucoRotationTracker(marker_id=marker_id, debug=debug, min_sharpness=min_sharpness, camera_index=aruco_camera_id, rate_hz=aruco_hz, units='rad')
		self.arucoDetector.start(wait=True, timeout=5.0)
		self.arucoDetector.reset_reward()

		# control box stuff
		self.box = SafeControlBox(max_pressure=self.max_pressure)
		if(not self.box.connect()):
			raise RuntimeError("Unable to connect to the controlbox, check the stuff and try again")

		# observation and action stuff
		self.action_space = spaces.Box( low=-1, high=1, shape=(3,), dtype=np.float32 )
		# observation can be 4 if contact sensor
		self.observation_space = spaces.Box( low=0, high=self.max_pressure, shape=(3,), dtype=np.float32 )
		self.current_prop = None
		self.current_img = None
		self.cropped_img = None
		self.target_size = 10
		self.current_step = 0
	
	def reset(self, seed=None, options=None):
		'''
		Reset the environment with a random target
		Args:
			seed: random seed (actually ignored)
			options: additional options (really not used)
		Returns:
			np.ndarray: Initial observation of the environment state.
		'''
		super().reset(seed=seed, options=options)
		self.box.reset()
		
		self.current_img, self.cropped_img = self.camera.get_clear_image(timeout=5.0)
		self.arucoDetector.reset_reward()
		self.current_prop = np.array([0.0, 0.0, 0.0])
		self.current_pressure = np.array([0.0, 0.0, 0.0])
		self.current_step = 0
		self.last_return = time.time()
		return self.current_prop, {}

	def step(self, action) -> tuple:
		'''
		Step in the REAL world
		action: action to take
		returns: observation (np.array), reward (float), terminated (bool), truncated (bool), info (dict)
		'''
		# do the action
		for i in range(3):
			self.current_pressure[i] = self.current_pressure[i] + 0.1*action[i]
			self.current_pressure[i] = min(self.max_pressure, max(self.current_pressure[i], 0))
		self.box.send_pressure_array(self.current_pressure)
		time.sleep(self.stepTime/2) # with this sleep the robot sees a bit the effect of the action
		self.current_prop = np.array([
			#self.pressure.safe_read(), 
			self.current_pressure[0], 
			self.current_pressure[1], 
			self.current_pressure[2]])
		self.current_img, self.cropped_img = self.camera.get_clear_image(timeout=self.stepTime/4)
		reward = self.arucoDetector.reset_reward()
		if reward > 0.03 or reward < -0.03: #ignore small rewards, they are probably noise
			reward *= self.reward_multiplier
		else:
			reward = 0
		info = {}
		elapsed_time = time.time() - self.last_return
		if elapsed_time < self.stepTime:
			time.sleep(self.stepTime - elapsed_time)
		else:
			print(f"Warning: step took longer than expected: {elapsed_time:.3f} seconds")
		self.last_return = time.time()
		if reward > 0.95: #TODO: after a full circle
			info['success'] = 1
		else:
			info['success'] = 0
		self.current_step += 1
		if(self.current_step > self.max_steps):
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
			cv2.imshow("Result", self.current_img)
			image = Image.fromarray(np.asarray(self.cropped_img))
			cropped_display = np.asarray(image.resize((512, 512), Image.NEAREST))
			cv2.imshow("Cropped", cropped_display)
			cv2.imshow("Aruco", self.arucoDetector.get_clear_image())
			cv2.waitKey(1)
			return self.current_img
		else:
			raise RuntimeError("Available render modes for the Real World: \{'rgb_array', 'human'\}")
		
	def close(self):
		cv2.destroyAllWindows()
		self.camera.stop()
		self.arucoDetector.stop()
		self.box.reset()
		self.box.disconnect()


if __name__ == "__main__":
	env = RealWorld(view_camera_id=1, width = 480, height = 480, cropped_width=64, cropped_height=64, camera_hz=20,
				  aruco_camera_id=0, marker_id=9, min_sharpness=100.0, aruco_hz=20,
				  render_mode='human', max_steps=100, env_hz=10, debug=True, rew_multiplier=8.0)
	observation, _ = env.reset()
	total_reward = 0
	done = False
	np.set_printoptions(precision=2, suppress=True)
	while not done:
		action = env.action_space.sample()
		observation, reward, terminated, truncated, info = env.step(action)
		#print(f"act: {action} rew:{reward:.2f}, obs:{observation}")
		print(f"rew:{reward:.2f}, obs:{observation}")
		env.render()
		done = terminated or truncated
		total_reward += reward
		if done:
			print(f"Game over! Total Reward: {total_reward}")
	env.close()