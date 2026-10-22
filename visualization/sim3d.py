"""Run the existing Python simulation with a local 3D browser viewer."""

import argparse
import json
import pickle
from dataclasses import asdict, replace
from http.server import ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from urllib.parse import urlsplit

import numpy as np

from civsim.actions import Action
from civsim.config import SimulationConfig
from civsim.observations import nearest_food
from civsim.world import World
from learning.baselines import heuristic_policy, random_policy
from visualization.model_preview import PreviewHandler
from experiments.diagnostics import movement_outcomes

STATIC = Path(__file__).with_suffix('')
CHECKPOINTS = STATIC.parents[1] / 'checkpoints'


def experiment_catalog():
    """One entry per experiment; retain training artifacts outside the menu."""
    from learning.inference import compatible_checkpoint
    entries = []
    for number in range(1, 15):
        stem = f'experiment_{number}'
        candidates = ([CHECKPOINTS / f'{stem}_best.pt'] if number >= 12 else [])
        candidates += [CHECKPOINTS / f'{stem}.pt', CHECKPOINTS / 'archive' / f'{stem}.pt']
        path = next((p for p in candidates if p.is_file()), None)
        available = path is not None and compatible_checkpoint(path, allow_legacy=True)
        reason = '' if available else ('older automatic-eating rules' if path else 'checkpoint missing')
        entries.append({'name': str(path.relative_to(CHECKPOINTS)) if path else f'{stem}.pt',
                        'label': f'Experiment {number}' + (f' — unavailable: {reason}' if reason else ''),
                        'available': available})
    # New carrying policies are a separate task, not extra stages of old experiments.
    from learning.inference import checkpoint_food_carrying
    for path in sorted(CHECKPOINTS.glob('*.pt')):
        if (compatible_checkpoint(path) and checkpoint_food_carrying(path)
                and not any(e['name'] == path.name for e in entries)):
            entries.append({'name': path.name, 'label': f'Food carrying · {path.stem}', 'available': True})
    return entries


def checkpoint_names():
    try:
        entries = experiment_catalog()
        from learning.inference import compatible_checkpoint
    except ImportError:
        return []
    # Preserve direct loading for tools/tests, while the browser uses the curated catalog.
    names = [entry['name'] for entry in entries if entry['available']]
    for p in sorted(CHECKPOINTS.glob('*.pt'), key=lambda p: p.stat().st_mtime, reverse=True):
        if (p.is_file() and p.resolve().parent == CHECKPOINTS.resolve()
                and p.name not in names and compatible_checkpoint(p, allow_legacy=True)):
            names.append(p.name)
    return names


class SimulationSession:
    def __init__(self):
        self.world = World(SimulationConfig())
        self.reset(123, 'heuristic')

    def reset(self, seed, policy, checkpoint=None, *, food_carrying=False):
        if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
            raise ValueError('Seed must be an integer between 0 and 4294967295.')
        if policy not in ('random', 'heuristic', 'ppo'):
            raise ValueError('Choose random, heuristic, or ppo policy.')
        if type(food_carrying) is not bool:
            raise ValueError('food_carrying must be a boolean')
        config = replace(self.world.config, food_carrying=food_carrying)
        learned = None
        if policy == 'ppo':
            names = checkpoint_names()
            checkpoint = checkpoint or (names[0] if names else None)
            if checkpoint not in names:
                raise ValueError('Choose an existing checkpoint from checkpoints/.')
            try:
                from learning.inference import LearnedPolicy, checkpoint_food_carrying
                config = replace(config, food_carrying=checkpoint_food_carrying(CHECKPOINTS / checkpoint))
                learned = LearnedPolicy(CHECKPOINTS / checkpoint, config, seed, allow_legacy=True)
            except (ImportError, OSError, RuntimeError, ValueError, KeyError, TypeError, EOFError, pickle.UnpicklingError) as exc:
                raise ValueError(f'Cannot load PPO checkpoint: {exc}') from exc
        self.seed, self.policy = seed, policy
        self.learned = learned
        self.checkpoint = checkpoint if policy == 'ppo' else None
        self.rng = np.random.default_rng(seed + 10000)
        self.world = World(config, automatic_eating=bool(learned and learned.automatic_eating))
        self.world.reset(seed)
        self.last_actions = {}
        self.last_outcomes = {}
        return self.state()

    def state(self):
        w = self.world
        alive = sum(a.alive for a in w.agents)
        finished = alive == 0 or (w.config.horizon is not None and w.tick >= w.config.horizon)
        probabilities = self.learned.probabilities(w) if self.learned and not finished else {}
        agents = []
        for agent in w.agents:
            row = asdict(agent)
            row['last_action'] = self.last_actions.get(agent.id)
            row['last_outcome'] = self.last_outcomes.get(agent.id)
            row['next_action_probabilities'] = {Action(i).name: probability
                for i, probability in enumerate(probabilities.get(agent.id, []))}
            agents.append(row)
        return {'config': asdict(w.config), 'seed': self.seed, 'policy': self.policy,
                'checkpoint': self.checkpoint,
                'eating_mode': 'automatic' if w.automatic_eating else 'explicit EAT',
                'action_masking': bool(self.learned and self.learned.action_masking),
                'tick': w.tick, 'agents': agents,
                'food': np.argwhere(w.food).tolist(), 'food_consumed': w.food_consumed,
                'food_energy_wasted': w.food_energy_wasted,
                'food_picked_up': w.food_picked_up,
                'finished': finished}

    def step(self):
        if (any(a.alive for a in self.world.agents) and
                (self.world.config.horizon is None or self.world.tick < self.world.config.horizon)):
            actions = {}
            if self.policy == 'ppo':
                actions = self.learned.actions(self.world)
            else:
                for a in self.world.agents:
                    if a.alive:
                        actions[a.id] = (random_policy(self.rng, food_carrying=self.world.config.food_carrying) if self.policy == 'random'
                                        else heuristic_policy(a.energy, nearest_food(self.world.food, a), self.rng,
                                            carried_food=a.carried_food if self.world.config.food_carrying else None))
            self.last_actions.update({agent_id: Action(action).name for agent_id, action in actions.items()})
            self.last_outcomes.update(movement_outcomes(self.world, actions))
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
            self.send_json({'checkpoints': experiment_catalog()})
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
                    result = self.server.session.reset(data.get('seed', 123), data.get('policy', 'heuristic'), data.get('checkpoint'), food_carrying=data.get('food_carrying', False))
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
    parser.add_argument('--food-carrying', action='store_true', help='Enable carrying for baseline policies')
    args = parser.parse_args()
    session = SimulationSession()
    if args.food_carrying:
        session.reset(123, 'heuristic', food_carrying=True)
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
