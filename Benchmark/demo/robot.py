'''
Thin robot backends for the demo: LiveRobot drives the real hardware, DryRobot does nothing
(for rehearsing the UI without the robot). Both expose the same small API used by the sequencer.
'''
import math
import os
import sys

import cv2
import numpy as np

sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))
import config as C


class DryRobot:
    live = False
    has_wheel = False

    def set_pressure(self, pressure):
        pass

    def start_episode(self):
        pass

    def progress_deg(self):
        return None  # sequencer falls back to the recorded progress

    def jpeg_main(self):
        return None

    def jpeg_wheel(self):
        return None

    def close(self):
        pass


class LiveRobot:
    live = True

    def __init__(self):
        # hardware imports only here, so the dry run doesn't need pyserial/cameras
        from envs.physical.control.safeControlBox import SafeControlBox
        from envs.physical.sense.CroppingCamera import CroppingCamera
        from envs.physical.sense.ArucoRotation import ArucoRotationTracker

        self.camera = None
        if C.FRONT_CAMERA_ID is not None:
            self.camera = CroppingCamera(camera_index=C.FRONT_CAMERA_ID, width=C.CAMERA_SIZE, height=C.CAMERA_SIZE,
                                         resized_width=64, resized_height=64, rate_hz=20)
            self.camera.start(wait=True, timeout=5.0)

        class TopCamera(ArucoRotationTracker):
            '''
            The tracker only publishes frames where the marker is detected and sharp, so as a video it
            goes blank or freezes. This keeps the latest frame anyway: the one with the gauge overlay
            when the marker is seen, the plain (cropped) camera frame otherwise.
            '''
            _latest = None

            def _process_frame(self, frame, t):
                ok = super()._process_frame(frame, t)
                with self._cond:
                    self._latest = self._image if ok else self._crop(frame)
                return ok

            def latest_frame(self):
                with self._cond:
                    return None if self._latest is None else self._latest.copy()

        # top camera: don't wait for the marker, so the demo starts even if it's hidden right now
        self.wheel = TopCamera(camera_index=C.TOP_CAMERA_ID, marker_id=C.ARUCO_MARKER_ID,
                               min_sharpness=C.ARUCO_MIN_SHARPNESS, rate_hz=20, units='rad',
                               overlay_target_deg=C.GOAL_DEG)
        self.wheel.start(wait=False)
        # with only the top camera it is the main video, otherwise it goes in the small inset
        self.has_wheel = self.camera is not None

        self.box = SafeControlBox(max_pressure=C.MAX_PRESSURE)
        if not self.box.connect():
            self.close()
            raise RuntimeError('Unable to connect to the control box')
        self.box.reset()
        self._progress = 0.0

    def set_pressure(self, pressure):
        p = np.clip(np.asarray(pressure, dtype=float), 0.0, C.MAX_PRESSURE)
        self.box.send_pressure_array(p)

    def start_episode(self):
        self._progress = 0.0
        if self.wheel is not None:
            self.wheel.reset_best()

    def progress_deg(self):
        '''rotation beyond the best angle, accumulated since start_episode (same as the env reward)'''
        if self.wheel.seconds_since_seen() == float('inf'):
            return None  # marker never seen: the sequencer shows the recorded rotation
        self._progress += self.wheel.reset_reward()
        return math.degrees(self._progress)

    def jpeg_main(self):
        '''main video: the front camera if connected, otherwise the top camera'''
        if self.camera is None:
            return self.jpeg_wheel()
        try:
            img, _ = self.camera.get_clear_image(timeout=1.0)
        except TimeoutError:
            return None
        return cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()

    def jpeg_wheel(self):
        img = self.wheel.latest_frame()
        if img is None:
            return None
        return cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()

    def close(self):
        if getattr(self, 'box', None) is not None and self.box.dev is not None:
            self.box.reset()
            self.box.disconnect()
        if getattr(self, 'camera', None) is not None:
            self.camera.stop()
        if getattr(self, 'wheel', None) is not None:
            self.wheel.stop()
