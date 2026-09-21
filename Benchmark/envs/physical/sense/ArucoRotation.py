import math
import threading
import time

import cv2
import numpy as np


class ArucoRotationTracker:
	"""
	Tracks the cumulative in-plane rotation of ONE ArUco marker on a background thread.

	The tracker owns the camera: a worker thread reads frames, detects the marker, unwraps
	its angle frame after frame and stores the result in shared variables. The public methods
	only read those variables (under a lock), so they return immediately and never touch the
	camera. Do not open cv2.VideoCapture elsewhere on the same device.

	Public API
		start() / stop()          start / stop the background thread (also usable with `with`)
		get_reward()              total rotation since the last reset_reward() (or since start())
		reset_reward()            returns the same value as get_reward() and re-zeroes it
		get_clear_image(...)      latest (cropped) frame in which the marker was detected
		seconds_since_seen()      how long ago the marker was last detected

	Sign convention: with ccw_positive=True, counter-clockwise rotation as seen in the image is positive.
	"""

	_UNITS = {'rad': 1.0, 'deg': 180.0 / math.pi, 'turns': 1.0 / (2.0 * math.pi)}

	def __init__(self, camera_index=0, width=640, height=480, rate_hz=20.0,
				 marker_id=None, units='rad', ccw_positive=True,
				 min_sharpness=None, brightness=150, debug=False):
		'''
		width, height: size of the center crop returned by get_clear_image (same crop as the env)
		rate_hz: processing rate cap; the real rate is also limited by the camera fps
		marker_id: if None exactly one marker must be visible, otherwise only this id is used
		units: 'rad', 'deg' or 'turns' for the reward
		min_sharpness: optional Laplacian-variance threshold on the marker area; frames below it are
			discarded (neither used for the angle nor returned as images). None disables the check.
		debug: draw the detected marker and its angle on the returned images
		'''
		if units not in self._UNITS:
			raise ValueError(f"units must be one of {list(self._UNITS)}")
		self.width = width
		self.height = height
		self.rate_hz = rate_hz
		self.marker_id = marker_id
		self.scale = self._UNITS[units]
		self.sign = -1.0 if ccw_positive else 1.0  # image y points down: atan2 is clockwise-positive
		self.min_sharpness = min_sharpness
		self.debug = debug

		self.cap = cv2.VideoCapture(camera_index)
		self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # the camera gives whatever it wants, we crop afterwards
		self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
		self.cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness)
		self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # may be ignored by some backends
		arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
		arucoParams = cv2.aruco.DetectorParameters()
		self.detector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)

		# shared state (written by the worker thread, read under the lock by everyone else)
		self._cond = threading.Condition()
		self._cum_angle = 0.0      # radians, unwrapped, accumulated since start()
		self._baseline = 0.0       # value of _cum_angle at the last reset_reward()
		self._image = None
		self._image_time = 0.0
		self._last_seen = None
		# worker-thread only
		self._last_angle = None

		self._stop_event = threading.Event()
		self._thread = None

	# ------------------------------------------------------------------ lifecycle
	def start(self, wait=True, timeout=50.0):
		'''Start the background thread. If wait=True, block until the marker has been seen once.'''
		if self._thread is not None:
			return self
		self._stop_event.clear()
		self._thread = threading.Thread(target=self._run, name="ArucoRotationTracker", daemon=True)
		self._thread.start()
		if wait:
			with self._cond:
				if not self._cond.wait_for(lambda: self._image is not None, timeout):
					self.stop()
					raise TimeoutError("ArUco marker not detected within the start timeout")
		return self

	def stop(self):
		self._stop_event.set()
		if self._thread is not None:
			self._thread.join(timeout=2.0)
			self._thread = None
		self.cap.release()

	def __enter__(self):
		return self.start()

	def __exit__(self, *exc):
		self.stop()

	# ------------------------------------------------------------------ public reads
	def get_reward(self) -> float:
		'''Total rotation since the last reset_reward() (or since start()). Returns immediately.'''
		with self._cond:
			return (self._cum_angle - self._baseline) * self.scale

	def reset_reward(self) -> float:
		'''Returns the reward accumulated so far and uses the current angle as the new zero.'''
		with self._cond:
			reward = (self._cum_angle - self._baseline) * self.scale
			self._baseline = self._cum_angle
			return reward

	def get_clear_image(self, newer_than=None, timeout=2.0):
		'''
		Returns a copy of the latest frame in which the marker was detected.
		newer_than: a time.time() value; if given, waits for a frame captured after it, which
			guarantees the image shows the effect of an action sent before that time.
		Raises TimeoutError if no such frame arrives within `timeout` seconds.
		'''
		def ready():
			return self._image is not None and (newer_than is None or self._image_time > newer_than)
		with self._cond:
			if not self._cond.wait_for(ready, timeout):
				raise TimeoutError("No image with a visible ArUco marker within the timeout")
			return self._image.copy()

	def seconds_since_seen(self) -> float:
		'''Time since the marker was last detected (inf if never). Large values mean tracking was lost.'''
		with self._cond:
			return float('inf') if self._last_seen is None else time.time() - self._last_seen

	# ------------------------------------------------------------------ worker thread
	def _run(self):
		period = 1.0 / self.rate_hz
		last_proc = 0.0
		while not self._stop_event.is_set():
			ok, frame = self.cap.read()  # blocks at camera fps, which also keeps the buffer fresh
			if not ok:
				time.sleep(0.01)
				continue
			t = time.time()
			if t - last_proc < 0.5 * period:  # camera faster than ~2x rate_hz: skip frames
				continue
			last_proc = t
			self._process_frame(frame, t)

	def _process_frame(self, frame, t) -> bool:
		img = self._crop(frame)
		corners, ids, _ = self.detector.detectMarkers(img)
		if ids is None or len(ids) == 0:
			return False
		ids = ids.flatten()
		if self.marker_id is not None:
			matches = np.where(ids == self.marker_id)[0]
			if len(matches) != 1:
				return False
			idx = int(matches[0])
		else:
			if len(ids) != 1:
				return False
			idx = 0
		pts = corners[idx].reshape(4, 2)
		if self.min_sharpness is not None and self._sharpness(img, pts) < self.min_sharpness:
			return False

		# marker orientation: average of the top edge and the bottom edge directions
		tl, tr, br, bl = pts
		v = (tr - tl) + (br - bl)
		angle = self.sign * math.atan2(v[1], v[0])
		delta = 0.0 if self._last_angle is None else self._wrap(angle - self._last_angle)
		self._last_angle = angle

		out = img
		if self.debug:
			out = img.copy()
			cv2.aruco.drawDetectedMarkers(out, [corners[idx]], ids[idx:idx + 1].reshape(-1, 1))
			cv2.putText(out, f"{math.degrees(angle):.0f} deg", (10, 25),
						cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

		with self._cond:
			self._cum_angle += delta
			self._image = out
			self._image_time = t
			self._last_seen = t
			self._cond.notify_all()
		return True

	# ------------------------------------------------------------------ helpers
	def _crop(self, img):
		h, w = img.shape[:2]
		x0 = max((w - self.width) // 2, 0)
		y0 = max((h - self.height) // 2, 0)
		return img[y0:y0 + self.height, x0:x0 + self.width]

	@staticmethod
	def _wrap(a):
		'''Wrap an angle difference to [-pi, pi).'''
		return (a + math.pi) % (2.0 * math.pi) - math.pi

	@staticmethod
	def _sharpness(img, pts):
		x, y, w, h = cv2.boundingRect(pts.astype(np.int32))
		roi = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[max(y, 0):y + h, max(x, 0):x + w]
		if roi.size == 0:
			return 0.0
		return cv2.Laplacian(roi, cv2.CV_64F).var()

if __name__ == "__main__":
	tracker = ArucoRotationTracker(marker_id=9, debug=True, min_sharpness=100.0, camera_index=0, rate_hz=20.0, units='rad')
	# 10 Hz = 0.1 seconds per iteration
	interval = 1.0 / 10.0 
	next_time = time.perf_counter()
	with tracker:
		while True:
			img = tracker.get_clear_image()
			rew = tracker.reset_reward()
			if rew >= 0.03 or rew <= -0.03:
				rew*=10
				print(f"Reward: {rew:.3f}, seconds since seen: {tracker.seconds_since_seen():.2f}")
			cv2.imshow("Aruco", img)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
			next_time += interval
			sleep_time = next_time - time.perf_counter()
			
			if sleep_time > 0:
				time.sleep(sleep_time)
			else:
				# If processing took longer than 0.1s, reset the clock to prevent 
				# the loop from firing rapidly to "catch up".
				next_time = time.perf_counter()