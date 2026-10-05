import cv2
import math
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
			  render_mode='rgb_array', max_steps=100, env_hz=10, debug=False, rew_multiplier=30.0):
		'''
		initialize the environment by doing important initialization stuff (in the real world)
		'''
		super(RealWorld, self).__init__()
		self.width = width
		self.height = height
		self.render_mode = render_mode
		self.reward_multiplier = rew_multiplier
		self.debug = debug
		self.max_pressure = 0.9
		if debug:
			self.max_pressure = 0.9
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
		# success = the marker has rotated at least half a turn (180 deg) since the episode
		# started; expressed in the tracker's own units so it stays correct if units != 'rad'.
		self.success_rotation_threshold = math.pi * self.arucoDetector.scale
		self.episode_start_rotation = self.arucoDetector.get_absolute_rotation()

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
		self.save = False
		self.target_size = 10
		self.current_step = 0
		self._windows_positioned = False

	def save_next_episode_video(self, id):
		'''
		Save the next episode to disk
		Args:
			id: id of the episode to save
		'''
		self.save = True
		self.run_id = id
		self.frame_original = []
		self.frame_cropped = []
		self.frame_aruco = []
		self.frame_pressure = []

	def save_now(self):
		if self.save:
			self.save = False
			fourcc = cv2.VideoWriter_fourcc(*'mp4v')
			
			# Helper function to dynamically save any frame list
			def write_video(filename, frames):
				if not frames:
					print(f"Warning: No frames to save for {filename}")
					return
				height, width = frames[0].shape[:2]
				is_color = len(frames[0].shape) == 3 and frames[0].shape[2] == 3
				out = cv2.VideoWriter(filename, fourcc, 1/self.stepTime, (width, height), isColor=is_color)
				for frame in frames:
					write_frame = np.array(frame, dtype=np.uint8)
					out.write(write_frame)
				out.release()
				print(f"Saved: {filename}")

			write_video(f'episode_{self.run_id}.mp4', self.frame_original)
			write_video(f'episode_{self.run_id}_cropped.mp4', self.frame_cropped)
			write_video(f'episode_{self.run_id}_aruco.mp4', self.frame_aruco)
			write_video(f'episode_{self.run_id}_pressure.mp4', self.frame_pressure)

			self.frame_original.clear()
			self.frame_cropped.clear()
			self.frame_aruco.clear()
			self.frame_pressure.clear()
	
	def render_pressure_window(self, width=220, height=320):
		'''
		Draws a small bar chart image showing how much pressure is currently sent to each of
		the 3 chambers, out of self.max_pressure. Used as the floating debug window in
		render(render_mode='human').
		'''
		img = np.full((height, width, 3), 30, dtype=np.uint8)
		margin = 30
		bar_area_h = height - 2 * margin
		bar_w = (width - 2 * margin) // 3 - 10
		colors = [(60, 180, 255), (80, 220, 100), (60, 120, 255)]  # BGR, one per chamber
		for i in range(3):
			x0 = margin + i * (bar_w + 10)
			x1 = x0 + bar_w
			y_top, y_bottom = margin, height - margin
			frac = float(np.clip(self.current_pressure[i] / self.max_pressure, 0.0, 1.0))
			y_fill = y_bottom - int(bar_area_h * frac)
			cv2.rectangle(img, (x0, y_top), (x1, y_bottom), (90, 90, 90), 1)
			cv2.rectangle(img, (x0, y_fill), (x1, y_bottom), colors[i], -1)
			cv2.putText(img, f"C{i + 1}", (x0, y_top - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
			cv2.putText(img, f"{self.current_pressure[i]:.2f}", (x0, y_bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
		return img

	def _layout_windows_once(self, window_rows):
		'''
		Positions each named window in a grid (one list of (name, width, height) per row) so
		they don't stack on top of each other, the first time render() is called in human mode.
		Args:
			window_rows: list of rows, each a list of (name, width, height) in display order
		'''
		if self._windows_positioned:
			return
		gap = 10
		y = 40
		for row in window_rows:
			x = 0
			for name, w, h in row:
				cv2.moveWindow(name, x, y)
				x += w + gap
			y += max(h for _, _, h in row) + gap
		self._windows_positioned = True

	def reset(self, seed=None, options=None):
		'''
		Reset the environment with a random target
		Args:
			seed: random seed (actually ignored)
			options: additional options (really not used)
		Returns:
			np.ndarray: Initial observation of the environment state.
		'''
		self.save_now()
		super().reset(seed=seed, options=options)
		self.box.reset()
		
		self.current_img, self.cropped_img = self.camera.get_clear_image(timeout=5.0)
		self.arucoDetector.reset_best()
		# new episode: re-baseline the success rotation odometer too, independently of the
		# per-step reward starting point reset_best() just set.
		self.episode_start_rotation = self.arucoDetector.get_absolute_rotation()
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
		# progress beyond the best angle reached so far this episode (never negative, see
		# ArucoRotationTracker.reset_reward), scaled by the multiplier
		reward = self.arucoDetector.reset_reward() * self.reward_multiplier

		info = {}
		elapsed_time = time.time() - self.last_return
		if elapsed_time < self.stepTime:
			time.sleep(self.stepTime - elapsed_time)
		else:
			print(f"Warning: step took longer than expected: {elapsed_time:.3f} seconds")
		self.last_return = time.time()
		# success once the marker has rotated at least half a turn (180 deg) since the episode
		# started - not the same thing as a single big per-step reward, which is why this reads
		# the tracker's odometer directly instead of thresholding `reward`.
		total_rotation = self.arucoDetector.get_absolute_rotation() - self.episode_start_rotation
		info['success'] = 1 if abs(total_rotation) >= self.success_rotation_threshold else 0
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
			aruco_display = self.arucoDetector.get_clear_image()
			cv2.imshow("Aruco", aruco_display)
			pressure_display = self.render_pressure_window()
			cv2.imshow("Pressure", pressure_display)
			self._layout_windows_once([
				[
					("Result", self.current_img.shape[1], self.current_img.shape[0]),
					("Cropped", cropped_display.shape[1], cropped_display.shape[0]),
					("Aruco", aruco_display.shape[1], aruco_display.shape[0]),
				],
				[
					("Pressure", pressure_display.shape[1], pressure_display.shape[0]),
				],
			])
			cv2.waitKey(1)
			if self.save:
				self.frame_original.append(self.current_img)
				self.frame_cropped.append(cropped_display)
				self.frame_aruco.append(aruco_display)
				self.frame_pressure.append(pressure_display)
			return self.current_img
		else:
			raise RuntimeError("Available render modes for the Real World: \{'rgb_array', 'human'\}")
		
	def close(self):
		self.save_now()
		cv2.destroyAllWindows()
		self.camera.stop()
		self.arucoDetector.stop()
		self.box.reset()
		self.box.disconnect()


if __name__ == "__main__":
	env = RealWorld(view_camera_id=1, width = 480, height = 480, cropped_width=64, cropped_height=64, camera_hz=20,
				  aruco_camera_id=0, marker_id=5, min_sharpness=100.0, aruco_hz=20,
				  render_mode='human', max_steps=100, env_hz=10, debug=True, rew_multiplier=8.0)
	observation, _ = env.reset()
	total_reward = 0
	done = False
	env.save_next_episode_video(2)
	np.set_printoptions(precision=2, suppress=True)

	rng = np.random.default_rng()

	# goal oriented blabbing
	# basically do lines
	# and then follow this lines adding some noise will make the robot explore more of the workspace
	# and possibly get reward since the reward is not dence but depends on teh contact with an gear
	LINE_NOISE_STD = 0.15          # exploration noise added on top of the line direction
	LINE_LEN_RANGE = (5, 15)       # steps a line segment lasts before a new target is picked
	RESET_STEPS = 5                # steps of action=-1 to bring pressure back to zero between lines

	def new_line_target():
		'''Random pressure setpoint (bar) to draw a straight line towards, in [0, max_pressure]^3.'''
		return rng.uniform(0.0, env.max_pressure, size=3).astype(np.float32)

	phase = 'draw'  # 'draw' -> follow a line, 'reset' -> retract back to zero pressure
	line_target = new_line_target()
	line_steps_left = rng.integers(*LINE_LEN_RANGE)
	reset_steps_left = 0

	while not done:
		if phase == 'draw' and line_steps_left <= 0:
			phase = 'reset'
			reset_steps_left = RESET_STEPS

		if phase == 'draw':
			line_steps_left -= 1
			# proportional step towards the target pressure (this traces a line in pressure space)
			# plus noise, so consecutive actions stay correlated instead of cancelling out like pure iid sampling
			direction = (line_target - env.current_pressure) / 0.1
			noise = rng.normal(0.0, LINE_NOISE_STD, size=3)
			action = np.clip(direction + noise, -1.0, 1.0).astype(np.float32)
		else:
			# retract to zero pressure before starting the next line
			reset_steps_left -= 1
			action = np.full(3, -1.0, dtype=np.float32)
			if reset_steps_left <= 0:
				phase = 'draw'
				line_target = new_line_target()
				line_steps_left = rng.integers(*LINE_LEN_RANGE)

		observation, reward, terminated, truncated, info = env.step(action)
		#print(f"act: {action} rew:{reward:.2f}, obs:{observation}")
		print(f"rew:{reward:.2f}, obs:{observation}")
		env.render()
		done = terminated or truncated
		total_reward += reward
		if done:
			print(f"Game over! Total Reward: {total_reward}")
	env.close()