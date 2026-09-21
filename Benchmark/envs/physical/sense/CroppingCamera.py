import math
import threading
import time

import cv2
import numpy as np

class CroppingCamera:
	"""
	Tracks the images at a given frame rate witha background thread.

	The tracker owns the camera: a worker thread reads frames, crops them and stores 
	the result in shared variables. The public methods only read those variables (under a lock), 
	so they return immediately and never touch the camera. 
	Do not open cv2.VideoCapture elsewhere on the same device.

	Public API
		start() / stop()          start / stop the background thread (also usable with `with`)
		get_clear_image(...)      latest frame with the original crop and the resized crop
	"""

	def __init__(self, camera_index=0, width=640, height=480, rate_hz=20.0,
			  	 resized_width=64, resized_height=64, brightness=150, debug=False):
		'''
		width, height: size of the center crop returned by get_clear_image (should be valid for the camera)
		rate_hz: processing rate cap; the real rate is also limited by the camera fps
		resized_width, resized_height: size of the image returned by get_clear_image (can be different from the crop size)
		brightness: camera brightness setting (0-255)
		debug: draw the detected marker and its angle on the returned images
		'''
		self.width = width
		self.height = height
		self.rate_hz = rate_hz
		self.resized_width = resized_width
		self.resized_height = resized_height
		self.debug = debug

		self.cap = cv2.VideoCapture(camera_index)
		self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # the camera gives whatever it wants, we crop afterwards
		self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
		self.cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness)
		self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # may be ignored by some backends

		# shared state (written by the worker thread, read under the lock by everyone else)
		self._cond = threading.Condition()
		self._image = None
		self._cropped_image = None
		self._image_time = 0.0
		self._last_seen = None
		self._stop_event = threading.Event()
		self._thread = None

	# ------------------------------------------------------------------ lifecycle
	def start(self, wait=True, timeout=5.0):
		'''Start the background thread. If wait=True, block until the marker has been seen once.'''
		if self._thread is not None:
			return self
		self._stop_event.clear()
		self._thread = threading.Thread(target=self._run, name="ImageCropper", daemon=True)
		self._thread.start()
		if wait:
			with self._cond:
				if not self._cond.wait_for(lambda: self._image is not None, timeout):
					self.stop()
					raise TimeoutError("Image not available within the start timeout")
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

	def get_clear_image(self, newer_than=None, timeout=2.0):
		'''
		Returns a copy of the latest frame and the cropped image.
		newer_than: a time.time() value; if given, waits for a frame captured after it, which
			guarantees the image shows the effect of an action sent before that time.
		Raises TimeoutError if no such frame arrives within `timeout` seconds.
		'''
		def ready():
			return self._image is not None and (newer_than is None or self._image_time > newer_than)
		with self._cond:
			if not self._cond.wait_for(ready, timeout):
				raise TimeoutError("No image with a available within the timeout")
			return self._image.copy(), self._cropped_image.copy()

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
		img = self._crop_standard(frame)
		cropped_img = self._crop(img)
		with self._cond:
			self._image = img
			self._image_time = t
			self._cropped_image = cropped_img
			self._cond.notify_all()
		return True

	# ------------------------------------------------------------------ helpers
	def _crop_standard(self, img):
		h, w = img.shape[:2]
		x0 = max((w - self.width) // 2, 0)
		y0 = max((h - self.height) // 2, 0)
		return img[y0:y0 + self.height, x0:x0 + self.width]

	def _crop(self, img):
		return cv2.resize(img, (self.resized_width, self.resized_height), interpolation=cv2.INTER_AREA)

if __name__ == "__main__":
	with CroppingCamera(camera_index=1, width=480, height=480, resized_width=64, resized_height=64, rate_hz=2.0, debug=True) as cam:
		while True:
			img, cropped_img = cam.get_clear_image()
			cv2.imshow("Original", img)
			cv2.imshow("Cropped", cropped_img)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				cv2.imwrite("Original.png", img)
				cv2.imwrite("Cropped.png", cropped_img)
				break