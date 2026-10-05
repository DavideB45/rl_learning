import math
import threading
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class ArucoRotationTracker:
	"""
	Tracks the cumulative in-plane rotation of ONE ArUco marker on a background thread.

	The tracker owns the camera: a worker thread reads frames, detects the marker, unwraps
	its angle frame after frame and stores the result in shared variables. The public methods
	only read those variables (under a lock), so they return immediately and never touch the
	camera. Do not open cv2.VideoCapture elsewhere on the same device.

	Public API
		start() / stop()          start / stop the background thread (also usable with `with`)
		get_reward()              how far the marker is beyond the best angle reached so far (>= 0)
		reset_reward()            returns the same value as get_reward() and raises the best angle to it
		reset_best()              uses the current angle as the new starting point / best (e.g. per episode)
		get_absolute_rotation()   total rotation since start(), never re-zeroed (an odometer)
		get_clear_image(...)      latest (cropped) frame in which the marker was detected
		seconds_since_seen()      how long ago the marker was last detected

	Sign convention: with ccw_positive=True, counter-clockwise rotation as seen in the image is positive.
	"""

	_UNITS = {'rad': 1.0, 'deg': 180.0 / math.pi, 'turns': 1.0 / (2.0 * math.pi)}

	def __init__(self, camera_index=0, width=640, height=480, rate_hz=20.0,
				 marker_id=None, units='rad', ccw_positive=True,
				 min_sharpness=None, brightness=150, debug=False,
				 overlay=True, overlay_radius=None, overlay_target_deg=90.0):
		'''
		width, height: size of the center crop returned by get_clear_image (same crop as the env)
		rate_hz: processing rate cap; the real rate is also limited by the camera fps
		marker_id: if None exactly one marker must be visible, otherwise only this id is used
		units: 'rad', 'deg' or 'turns' for the reward
		min_sharpness: optional Laplacian-variance threshold on the marker area; frames below it are
			discarded (neither used for the angle nor returned as images). None disables the check.
		debug: draw the detected marker and its angle on the returned images
		overlay: draw a progress gauge around the marker on the returned images: the arc goes from
			the orientation at the last reset_best() to the best angle reached, the dot is the current
			orientation and the label is the progress in degrees
		overlay_radius: gauge radius in pixels; None sizes it from the marker
		overlay_target_deg: where to draw the target tick (and turn the arc green); None hides it
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
		self.overlay = overlay
		self.overlay_radius = overlay_radius
		self.overlay_target_deg = overlay_target_deg

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
		self._best = 0.0           # highest _cum_angle already rewarded since the last reset_best()
		self._start = 0.0          # _cum_angle at the last reset_best(), only used by the overlay
		self._image = None
		self._image_time = 0.0
		self._last_seen = None
		# worker-thread only
		self._last_angle = None
		self._ov_center = None     # overlay values smoothed over frames, so the gauge doesn't jitter
		self._ov_radius = None
		self._ov_progress = 0.0
		self._fonts = {}

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
		'''
		Progress beyond the best angle reached so far: max(0, current - best). Rotating back
		and then forward again over already-covered ground gives nothing, only new ground pays.
		Does not move the best angle. Returns immediately.
		'''
		with self._cond:
			return max(0.0, self._cum_angle - self._best) * self.scale

	def reset_reward(self) -> float:
		'''Returns the same value as get_reward() and, if the current angle beats the best, makes it the new best.'''
		with self._cond:
			reward = max(0.0, self._cum_angle - self._best) * self.scale
			self._best = max(self._best, self._cum_angle)
			return reward

	def reset_best(self):
		'''Uses the current angle as the new starting point (and best), e.g. at the start of an episode.'''
		with self._cond:
			self._best = self._cum_angle
			self._start = self._cum_angle

	def get_absolute_rotation(self) -> float:
		'''
		Total unwrapped rotation since start(), in the configured units. Unlike get_reward()/
		reset_reward(), this is never re-zeroed, so it works as a stable odometer: a caller can
		snapshot it (e.g. at episode reset) and diff against it later to measure rotation over
		an arbitrary span, without disturbing the best angle those two methods use.
		'''
		with self._cond:
			return self._cum_angle * self.scale

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

		if self.overlay:
			with self._cond:
				cum = self._cum_angle + delta
				best, start = self._best, self._start
			if out is img:
				out = img.copy()
			out = self._draw_overlay(out, pts, angle, cum, best, start)

		with self._cond:
			self._cum_angle += delta
			self._image = out
			self._image_time = t
			self._last_seen = t
			self._cond.notify_all()
		return True

	# ------------------------------------------------------------------ overlay
	_TRACK = (255, 255, 255)
	_ARC = (36, 165, 245)        # amber (BGR)
	_ARC_DONE = (94, 197, 34)    # green (BGR), once the target is reached
	_SHADOW = (20, 20, 20)

	def _draw_overlay(self, img, pts, angle, cum, best, start):
		'''
		Progress gauge centred on the marker. Angles are converted back to image space (clockwise,
		y down, as cv2.ellipse wants them) so the dot physically follows the marker's edge.
		'''
		center = pts.mean(axis=0)
		side = float(np.mean([np.linalg.norm(pts[i] - pts[(i + 1) % 4]) for i in range(4)]))
		radius = self.overlay_radius if self.overlay_radius is not None else 1.15 * side
		if self._ov_center is None:
			self._ov_center, self._ov_radius = center, radius
		else:
			self._ov_center = 0.7 * self._ov_center + 0.3 * center
			self._ov_radius = 0.9 * self._ov_radius + 0.1 * radius
		# best including the current frame, so the gauge doesn't wait for the next reset_reward()
		progress = max(0.0, max(best, cum) - start)
		self._ov_progress += 0.35 * (progress - self._ov_progress)  # ease towards the true value

		# `angle` and the cumulative values use the sign convention; image angles are raw atan2
		now_img = math.degrees(self.sign * angle)
		start_img = now_img - math.degrees(self.sign * (cum - start))
		end_img = start_img + math.degrees(self.sign * min(self._ov_progress, 2.0 * math.pi))
		done = (self.overlay_target_deg is not None
				and math.degrees(progress) >= self.overlay_target_deg)

		SHIFT = 4  # sub-pixel precision for smooth, anti-aliased shapes
		k = 1 << SHIFT
		cx, cy = self._ov_center
		r = self._ov_radius
		thick = max(4, int(round(r * 0.14)))
		c = (int(round(cx * k)), int(round(cy * k)))
		ax = (int(round(r * k)), int(round(r * k)))

		def on_ring(deg, rr=r):
			return (cx + rr * math.cos(math.radians(deg)), cy + rr * math.sin(math.radians(deg)))

		def fixed(p):
			return (int(round(p[0] * k)), int(round(p[1] * k)))

		def arc(canvas, a0, a1, color, t):
			a0, a1 = min(a0, a1), max(a0, a1)
			cv2.ellipse(canvas, c, ax, 0, a0, a1, color, t, cv2.LINE_AA, SHIFT)
			for a in (a0, a1):  # rounded caps
				cv2.circle(canvas, fixed(on_ring(a)), (t // 2) * k, color, -1, cv2.LINE_AA, SHIFT)

		# translucent layer: track ring, shadow under the arc, start / target ticks
		layer = img.copy()
		cv2.circle(layer, c, ax[0], self._TRACK, max(2, thick // 2), cv2.LINE_AA, SHIFT)
		if self._ov_progress > 1e-3:
			arc(layer, start_img, end_img, self._SHADOW, thick + 4)
		ticks = [start_img]
		if self.overlay_target_deg is not None:
			ticks.append(start_img + self.sign * self.overlay_target_deg)
		for a in ticks:
			cv2.line(layer, fixed(on_ring(a, r - thick)), fixed(on_ring(a, r + thick)),
					 self._TRACK, 2, cv2.LINE_AA, SHIFT)
		img = cv2.addWeighted(layer, 0.45, img, 0.55, 0)

		# opaque layer: progress arc and current-orientation dot
		if self._ov_progress > 1e-3:
			arc(img, start_img, end_img, self._ARC_DONE if done else self._ARC, thick)
		dot = fixed(on_ring(now_img))
		cv2.circle(img, dot, (thick // 2 + 3) * k, self._SHADOW, -1, cv2.LINE_AA, SHIFT)
		cv2.circle(img, dot, (thick // 2 + 1) * k, (255, 255, 255), -1, cv2.LINE_AA, SHIFT)

		label = f"{math.degrees(progress):.0f}\u00b0"
		if self.overlay_target_deg is not None:
			label += f" / {self.overlay_target_deg:.0f}\u00b0"
		return self._draw_label(img, label, (cx, cy + r + thick + 8), done)

	def _font(self, size):
		if size not in self._fonts:
			font = None
			for path in ("/System/Library/Fonts/Avenir Next.ttc", "/System/Library/Fonts/HelveticaNeue.ttc",
						 "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
				try:
					font = ImageFont.truetype(path, size, index=0)
					break
				except OSError:
					continue
			self._fonts[size] = font if font is not None else ImageFont.load_default(size)
		return self._fonts[size]

	def _draw_label(self, img, text, anchor, done):
		'''Text on a rounded translucent pill, centred under `anchor` (moved above the gauge if it doesn't fit).'''
		font = self._font(max(14, int(round(self._ov_radius * 0.32))))
		h_img, w_img = img.shape[:2]
		x0, y0, x1, y1 = font.getbbox(text)
		tw, th = x1 - x0, y1 - y0
		pad_x, pad_y = th * 0.7, th * 0.45
		w, h = tw + 2 * pad_x, th + 2 * pad_y
		left = min(max(anchor[0] - w / 2, 4), w_img - w - 4)
		top = anchor[1]
		if top + h > h_img - 4:  # no room below: put it above the gauge
			top = 2 * self._ov_center[1] - anchor[1] - h
		top = min(max(top, 4), h_img - h - 4)

		base = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).convert("RGBA")
		layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
		draw = ImageDraw.Draw(layer)
		accent = self._ARC_DONE if done else self._ARC
		draw.rounded_rectangle((left, top, left + w, top + h), radius=h / 2,
							   fill=(20, 20, 20, 170), outline=accent[::-1] + (255,), width=2)
		draw.text((left + pad_x - x0, top + pad_y - y0), text, font=font, fill=(255, 255, 255, 255))
		out = Image.alpha_composite(base, layer).convert("RGB")
		return cv2.cvtColor(np.asarray(out), cv2.COLOR_RGB2BGR)

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
	tracker = ArucoRotationTracker(marker_id=5, debug=True, min_sharpness=100.0, camera_index=0, rate_hz=20.0, units='rad')
	# 10 Hz = 0.1 seconds per iteration
	interval = 1.0 / 10.0 
	next_time = time.perf_counter()
	tot_reward = 0.0
	with tracker:
		i = 0
		while True and i < 150:
			i+= 1
			img = tracker.get_clear_image()
			rew = tracker.reset_reward()
			rew*=30
			tot_reward += rew
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
	print(f"Total Reward: {tot_reward:.3f}")