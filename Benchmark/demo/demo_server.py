'''
Lab demo for kids: shows how the robot "learns" by replaying three recorded episodes
(early / middle / final training round) on the real robot, with "dreams" in between made of
other recorded episodes. No model is used, it's all a replay of the dataset.

Usage (from the repo root, with the rl_env environment):
    python demo/demo_server.py            # real robot + top camera
    python demo/demo_server.py --dry-run  # no hardware, UI only (plays recorded frames)
Then press the big START button in the browser page that opens (F11 / ctrl+cmd+F for fullscreen).
Space bar = "Avanti" when WAIT_FOR_CLICK is on, Esc = emergency stop (deflates everything).

Settings (which episodes, timings, camera ids...) are in demo/config.py.
'''
import argparse
import json
import math
import os
import sys
import threading
import time
import webbrowser
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

import cv2
import numpy as np

sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))
sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))
import config as C
from robot import DryRobot, LiveRobot


def load_dataset():
    with open(os.path.join(C.DATA_DIR, C.TRANSITIONS)) as f:
        data = json.load(f)
    episodes = []
    for ep in range(len(data['reward'])):
        pressures = np.clip(np.asarray(data['proprioception'][ep], dtype=float), 0.0, C.MAX_PRESSURE)
        # progress in degrees after each step, aligned with pressures (pressures[0] is the start)
        progress = np.concatenate([[0.0], np.cumsum(data['reward'][ep])]) / C.REW_MULTIPLIER
        progress = np.degrees(progress)[:len(pressures)]
        episodes.append({'pressures': pressures, 'progress': progress})
    return episodes


def episode_info(episodes, ep, rnd):
    e = episodes[ep]
    return {
        'ep': ep,
        'round': rnd,
        'n_steps': len(e['pressures']),
        'final_deg': round(float(e['progress'][-1]), 1),
        'pressures': np.round(e['pressures'], 3).tolist(),
        'progress': np.round(e['progress'], 1).tolist(),
    }


class Show:
    '''Runs the sequence of phases on a background thread and exposes its state to the UI.'''

    def __init__(self, robot, episodes):
        self.robot = robot
        self.episodes = episodes
        self.tries = [episode_info(episodes, C.round_to_episode(r), r) for r in C.TRY_ROUNDS]
        # each dream: episodes at evenly spaced quantiles of the rotation reached in its round
        # range, worst to best, so the dream goes from failing to doing well
        self.dreams = []
        for lo, hi in C.DREAM_ROUND_RANGES:
            rounds = sorted(range(lo, hi + 1), key=lambda r: episodes[C.round_to_episode(r)]['progress'][-1])
            picks = np.unique(np.linspace(0, len(rounds) - 1, min(C.DREAM_CLIPS, len(rounds))).round().astype(int))
            self.dreams.append([episode_info(episodes, C.round_to_episode(rounds[k]), rounds[k]) for k in picks])

        self.lock = threading.Lock()
        self.hw_lock = threading.Lock()  # serial writes come from the sequencer and from /stop
        self.abort = threading.Event()
        self.next = threading.Event()
        self.thread = None
        self.state = self._idle_state()

    # ------------------------------------------------------------------ state
    def _idle_state(self):
        return {'phase': 'idle', 'phase_id': 0, 'idx': 0, 'phase_started': time.time(), 'duration': None,
                'step': 0, 'pressure': [0.0, 0.0, 0.0], 'progress_deg': 0.0, 'results': [None] * len(self.tries),
                'waiting_click': False}

    def get_state(self):
        with self.lock:
            s = dict(self.state)
        s['now'] = time.time()
        return s

    def _set(self, **kw):
        with self.lock:
            self.state.update(kw)

    def _phase(self, phase, idx, duration=None, waiting_click=False):
        with self.lock:
            self.state.update(phase=phase, idx=idx, phase_id=self.state['phase_id'] + 1,
                              phase_started=time.time(), duration=duration, waiting_click=waiting_click)

    def info(self):
        main_view = 'recorded' if not self.robot.live else 'top' if self.robot.camera is None else 'front'
        return {'mode': 'live' if self.robot.live else 'dry', 'has_wheel': self.robot.has_wheel, 'main_view': main_view,
                'tries': self.tries, 'dreams': self.dreams, 'goal_deg': C.GOAL_DEG,
                'dream_fps': C.DREAM_FPS, 'dream_clip_pause': C.DREAM_CLIP_PAUSE,
                'replay_hz': C.REPLAY_HZ, 'max_pressure': C.MAX_PRESSURE}

    # ------------------------------------------------------------------ controls
    def start(self):
        if self.thread is not None and self.thread.is_alive():
            return False
        self.abort.clear()
        self.next.clear()
        with self.lock:
            phase_id = self.state['phase_id']
            self.state = self._idle_state()
            self.state['phase_id'] = phase_id
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return True

    def stop(self):
        self.abort.set()
        self._send_pressure([0.0, 0.0, 0.0])
        if self.thread is not None:
            self.thread.join(timeout=3.0)
        self._send_pressure([0.0, 0.0, 0.0])
        self._phase('idle', 0)

    def advance(self):
        self.next.set()

    def _send_pressure(self, p):
        with self.hw_lock:
            self.robot.set_pressure(p)
        self._set(pressure=[float(x) for x in p])

    # ------------------------------------------------------------------ sequence
    def _wait(self, seconds, clickable=False):
        '''sleep for `seconds` (or until "Avanti" if clickable and WAIT_FOR_CLICK); False if aborted'''
        self.next.clear()
        if clickable and C.WAIT_FOR_CLICK:
            while not self.next.is_set():
                if self.abort.wait(0.05):
                    return False
            return True
        return not self.abort.wait(seconds)

    def _run(self):
        n = len(self.tries)
        for i in range(n):
            self._phase('intro', i, None if C.WAIT_FOR_CLICK else C.INTRO_TIME, C.WAIT_FOR_CLICK)
            if not self._wait(C.INTRO_TIME, clickable=True):
                return
            if not self._replay(i):
                return
            self._send_pressure([0.0, 0.0, 0.0])
            self._phase('result', i, None if C.WAIT_FOR_CLICK else C.RESULT_TIME, C.WAIT_FOR_CLICK)
            if not self._wait(C.RESULT_TIME, clickable=True):
                return
            if i < len(self.dreams) and i < n - 1:
                clips = self.dreams[i]
                duration = sum(c['n_steps'] / C.DREAM_FPS + C.DREAM_CLIP_PAUSE for c in clips)
                self._phase('dream', i, duration)
                if not self._wait(duration):
                    return
        self._phase('finale', n - 1)

    def _replay(self, i):
        t = self.tries[i]
        pressures = self.episodes[t['ep']]['pressures']
        recorded = self.episodes[t['ep']]['progress']
        self._set(step=0, progress_deg=0.0)
        self._phase('try', i, len(pressures) / C.REPLAY_HZ + C.HOLD_AFTER_REPLAY)
        self.robot.start_episode()
        period = 1.0 / C.REPLAY_HZ
        next_t = time.time()
        progress = 0.0
        for k, p in enumerate(pressures):
            if self.abort.is_set():
                return False
            self._send_pressure(p)
            live = self.robot.progress_deg()
            progress = float(recorded[k]) if live is None else live
            self._set(step=k, progress_deg=progress)
            next_t += period
            time.sleep(max(0.0, next_t - time.time()))
        if not self._wait(C.HOLD_AFTER_REPLAY):
            return False
        live = self.robot.progress_deg()
        progress = progress if live is None else live
        with self.lock:
            self.state['results'][i] = round(progress, 1)
            self.state['progress_deg'] = progress
        return True


# ---------------------------------------------------------------------- http
class Handler(BaseHTTPRequestHandler):
    show = None
    sprites = {}
    sprite_lock = threading.Lock()

    def log_message(self, fmt, *args):
        pass

    def _send(self, body, ctype, code=200):
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')  # always fresh, it's all local anyway
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj):
        self._send(json.dumps(obj).encode(), 'application/json')

    def _static(self, name):
        path = os.path.normpath(os.path.join(C.STATIC_DIR, name))
        if not path.startswith(C.STATIC_DIR) or not os.path.isfile(path):
            return self._send(b'not found', 'text/plain', 404)
        ctype = {'.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8',
                 '.css': 'text/css; charset=utf-8', '.svg': 'image/svg+xml'}.get(os.path.splitext(path)[1],
                                                                                'application/octet-stream')
        with open(path, 'rb') as f:
            body = f.read()
        self.send_response(200)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def _sprite(self, ep):
        '''all the frames of an episode side by side in one png (img_<ep>_<step>.png in DATA_DIR)'''
        with self.sprite_lock:
            if ep not in self.sprites:
                n = len(self.show.episodes[ep]['pressures'])
                frames = []
                for s in range(n):
                    # the dataset pngs were saved by PIL straight from the BGR camera frames, so
                    # reading them with cv2 gives RGB: swap back to BGR before re-encoding
                    img = cv2.imread(os.path.join(C.DATA_DIR, f'img_{ep}_{s}.png'))
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    else:
                        img = frames[-1] if frames else np.zeros((64, 64, 3), np.uint8)
                    frames.append(img)
                self.sprites[ep] = cv2.imencode('.png', np.hstack(frames))[1].tobytes()
            body = self.sprites[ep]
        self._send(body, 'image/png')

    def _mjpeg(self, grab):
        self.send_response(200)
        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        try:
            while True:
                jpg = grab()
                if jpg is not None:
                    self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\n')
                    self.wfile.write(f'Content-Length: {len(jpg)}\r\n\r\n'.encode())
                    self.wfile.write(jpg + b'\r\n')
                time.sleep(1 / 20)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        path = self.path.split('?')[0]
        if path == '/':
            return self._static('index.html')
        if path.startswith('/static/'):
            return self._static(path[len('/static/'):])
        if path == '/info':
            return self._json(self.show.info())
        if path == '/state':
            return self._json(self.show.get_state())
        if path.startswith('/sprite/') and path.endswith('.png'):
            try:
                ep = int(path[len('/sprite/'):-4])
            except ValueError:
                ep = -1
            if 0 <= ep < len(self.show.episodes):
                return self._sprite(ep)
        if path == '/video.mjpg' and self.show.robot.live:
            return self._mjpeg(self.show.robot.jpeg_main)
        if path == '/wheel.mjpg' and self.show.robot.has_wheel:
            return self._mjpeg(self.show.robot.jpeg_wheel)
        self._send(b'not found', 'text/plain', 404)

    def do_POST(self):
        path = self.path.split('?')[0]
        if path == '/start':
            return self._json({'ok': self.show.start()})
        if path == '/stop':
            self.show.stop()
            return self._json({'ok': True})
        if path == '/next':
            self.show.advance()
            return self._json({'ok': True})
        self._send(b'not found', 'text/plain', 404)


def main():
    parser = argparse.ArgumentParser(description='Kids demo: replay of recorded episodes with a friendly UI')
    parser.add_argument('--dry-run', action='store_true', help='no hardware, the UI plays recorded frames')
    parser.add_argument('--port', type=int, default=C.PORT)
    parser.add_argument('--no-browser', action='store_true', help='do not open the browser automatically')
    args = parser.parse_args()

    episodes = load_dataset()
    robot = DryRobot() if args.dry_run else LiveRobot()
    show = Show(robot, episodes)
    for t in show.tries:
        print(f"[demo] try: round {t['round']:>2} (episode {t['ep']}), recorded rotation {t['final_deg']:.0f} deg")

    Handler.show = show
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    server.daemon_threads = True
    url = f'http://127.0.0.1:{args.port}/'
    print(f'[demo] open {url}  (ctrl+c to quit)')
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\n[demo] stopping, deflating the robot...')
    finally:
        show.abort.set()
        with show.hw_lock:
            robot.set_pressure([0.0, 0.0, 0.0])
        robot.close()
        server.server_close()


if __name__ == '__main__':
    main()
