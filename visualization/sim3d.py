"""Run the existing Python simulation with a local 3D browser viewer."""

import argparse
import json
import pickle
from dataclasses import asdict
from http.server import ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from urllib.parse import urlsplit

import numpy as np

from civsim.config import SimulationConfig
from civsim.observations import nearest_food
from civsim.world import World
from learning.baselines import heuristic_policy, random_policy
from visualization.model_preview import PreviewHandler

STATIC = Path(__file__).with_suffix('')
CHECKPOINTS = STATIC.parents[1] / 'checkpoints'


def checkpoint_names():
    return [p.name for p in sorted(CHECKPOINTS.glob('*.pt'),
                                  key=lambda p: p.stat().st_mtime, reverse=True)
            if p.is_file() and p.resolve().parent == CHECKPOINTS.resolve()]


class SimulationSession:
    def __init__(self):
        self.world = World(SimulationConfig())
        self.reset(123, 'heuristic')

    def reset(self, seed, policy, checkpoint=None):
        if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
            raise ValueError('Seed must be an integer between 0 and 4294967295.')
        if policy not in ('random', 'heuristic', 'ppo'):
            raise ValueError('Choose random, heuristic, or ppo policy.')
        learned = None
        if policy == 'ppo':
            names = checkpoint_names()
            checkpoint = checkpoint or (names[0] if names else None)
            if checkpoint not in names:
                raise ValueError('Choose an existing checkpoint from checkpoints/.')
            try:
                from learning.inference import LearnedPolicy
                learned = LearnedPolicy(CHECKPOINTS / checkpoint, self.world.config, seed)
            except (ImportError, OSError, RuntimeError, ValueError, KeyError, TypeError, EOFError, pickle.UnpicklingError) as exc:
                raise ValueError(f'Cannot load PPO checkpoint: {exc}') from exc
        self.seed, self.policy = seed, policy
        self.learned = learned
        self.checkpoint = checkpoint if policy == 'ppo' else None
        self.rng = np.random.default_rng(seed + 10000)
        self.world.reset(seed)
        return self.state()

    def state(self):
        w = self.world
        alive = sum(a.alive for a in w.agents)
        return {'config': asdict(w.config), 'seed': self.seed, 'policy': self.policy,
                'checkpoint': self.checkpoint,
                'tick': w.tick, 'agents': [asdict(a) for a in w.agents],
                'food': np.argwhere(w.food).tolist(), 'food_consumed': w.food_consumed,
                'finished': alive == 0 or (w.config.horizon is not None and w.tick >= w.config.horizon)}

    def step(self):
        if not self.state()['finished']:
            actions = {}
            if self.policy == 'ppo':
                actions = self.learned.actions(self.world)
            else:
                for a in self.world.agents:
                    if a.alive:
                        actions[a.id] = (random_policy(self.rng) if self.policy == 'random'
                                        else heuristic_policy(a.energy, nearest_food(self.world.food, a), self.rng))
            self.world.step(actions)
        return self.state()


class SimulationHandler(PreviewHandler):
    def send_json(self, data, code=200):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == '/api/checkpoints':
            self.send_json({'checkpoints': checkpoint_names()})
        elif path == '/api/state':
            with self.server.session_lock:
                self.send_json(self.server.session.state())
        elif path in ('/', '/view.js', '/assets.js'):
            name = {'/': 'index.html', '/view.js': 'view.js', '/assets.js': 'assets.js'}[path]
            data = (STATIC / name).read_bytes()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html' if path == '/' else 'text/javascript')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        else:
            super().do_GET()

    def do_POST(self):
        # State changes accept only same-origin JSON requests.
        origin = self.headers.get('Origin')
        if origin and origin != f"http://{self.headers.get('Host')}":
            self.send_json({'error': 'Origin not allowed'}, 403)
            return
        if self.headers.get('Content-Type') != 'application/json':
            self.send_json({'error': 'Expected application/json'}, 400)
            return
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 1024:
                raise ValueError('Invalid request size')
            data = json.loads(self.rfile.read(length))
            if not isinstance(data, dict):
                raise ValueError('Expected a JSON object')
            with self.server.session_lock:
                path = urlsplit(self.path).path
                if path == '/api/step':
                    result = self.server.session.step()
                elif path == '/api/reset':
                    result = self.server.session.reset(data.get('seed', 123), data.get('policy', 'heuristic'), data.get('checkpoint'))
                else:
                    self.send_json({'error': 'Unknown endpoint'}, 404)
                    return
            self.send_json(result)
        except (ValueError, TypeError) as exc:
            self.send_json({'error': str(exc)}, 400)

    def log_message(self, format, *args):
        # Avoid printing one line for every simulation tick.
        if len(args) > 1 and str(args[1]) != '200':
            super().log_message(format, *args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8001)
    parser.add_argument('--checkpoint', help='Checkpoint filename/path within checkpoints/, or latest')
    args = parser.parse_args()
    session = SimulationSession()
    if args.checkpoint:
        names = checkpoint_names()
        name = (names[0] if names else '') if args.checkpoint == 'latest' else Path(args.checkpoint).name
        try:
            session.reset(123, 'ppo', name)
        except ValueError as exc:
            parser.error(str(exc))
    with ThreadingHTTPServer(('127.0.0.1', args.port), SimulationHandler) as server:
        server.session = session
        server.session_lock = Lock()
        print(f'Open http://127.0.0.1:{args.port} — Ctrl+C to stop.', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
