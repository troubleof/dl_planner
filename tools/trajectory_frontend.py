"""Small native-range CNN proposing geometry, never executable controls.

YOPO-inspired multi-candidate output, trained from this project's EGO teachers.
Every proposal still goes through the unchanged EGO/backup/commit pipeline.
"""
import hashlib
import time
from pathlib import Path
import numpy as np
from navigation_common import CONFIG, save
from adapted_frontends import inputs, primitives, rotation

FORMAT = 'range_trajectory_v1'
DELTA_SCALE = np.array([3., .75, .30], dtype=np.float32)
FRACTIONS = np.array([1/3, 2/3, 1.], dtype=np.float32)


def geometry(request):
    start = np.asarray(request['start_state']['p'], dtype=float)
    anchors, frames = [], []
    for route in primitives(request):
        end = np.asarray(route['points'][-1])
        frame, _ = rotation(end, start)
        anchors.append(start + FRACTIONS[:, None] * (end-start))
        frames.append(frame)
    return np.asarray(anchors), np.asarray(frames)


def model_class():
    import torch
    class TrajectoryNetwork(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Sequential(
                torch.nn.Conv2d(3,12,3,stride=(1,2),padding=1), torch.nn.ReLU(),
                torch.nn.Conv2d(12,24,3,stride=2,padding=1), torch.nn.ReLU(),
                torch.nn.AdaptiveAvgPool2d((3,9)), torch.nn.Flatten())
            self.fusion = torch.nn.Sequential(torch.nn.Linear(648+22,64),torch.nn.ReLU())
            self.score_head = torch.nn.Linear(64,32)
            self.trajectory_head = torch.nn.Linear(64,15*3*3)
            torch.nn.init.zeros_(self.trajectory_head.weight)
            torch.nn.init.zeros_(self.trajectory_head.bias)

        def forward(self, image, state):
            latent = self.fusion(torch.cat((self.encoder(image),state),dim=1))
            scores = self.score_head(latent)
            offsets = torch.tanh(self.trajectory_head(latent)).reshape(-1,15,3,3)
            return scores[:,:16], scores[:,16:], offsets
    return TrajectoryNetwork


def decode(request, score, feasible, offsets, fixed=False):
    """Deterministic finite geometry proposals; repair is action 15."""
    anchors, frames = geometry(request)
    start = np.asarray(request['start_state']['p'], dtype=float)
    utility = np.asarray(score) - 2*(1-np.asarray(feasible))
    points = anchors if fixed else anchors + np.einsum('nkj,nij->nki',np.asarray(offsets)*DELTA_SCALE,frames)
    routes = []
    for index in np.argsort(-utility, kind='stable'):
        if index == 15:
            break
        p = points[index]
        local = (p-start) @ frames[index]
        # Reject degenerate/backtracking network geometry, not map validation.
        if not np.isfinite(p).all() or np.any(np.diff(np.r_[0.,local[:,0]]) <= .01):
            continue
        routes.append(dict(id=int(index),score=float(-utility[index]),points=[start.tolist(),*p.tolist()]))
        if len(routes) == 3:
            break
    return routes, utility


class TrajectoryFrontend:
    def __init__(self, weight, fixed=False):
        import torch
        self.torch = torch
        torch.set_num_threads(1)
        self.model = model_class()().eval()
        checkpoint = torch.load(weight,map_location='cpu',weights_only=True)
        if checkpoint.get('format') != FORMAT:
            raise ValueError('unsupported trajectory model format')
        self.model.load_state_dict(checkpoint['state_dict'])
        self.fixed = fixed
        self.metadata = dict(frontend=FORMAT, fixed_geometry=fixed,
            weight=str(weight),sha256=hashlib.sha256(Path(weight).read_bytes()).hexdigest(),
            parameters=sum(p.numel() for p in self.model.parameters()),seed=checkpoint['seed'],
            input_shape=[3,17,72],state_dim=22,candidate_count=15,waypoints=3,
            repair_action=15,maximum_proposals=3,extra_delay_assumption_s=.08,
            safety='Original EGO/whole-curve/backup/commit checks remain mandatory')
        for _ in range(5):
            with torch.inference_mode():
                self.model(torch.zeros(1,3,17,72),torch.zeros(1,22))

    def propose(self, cloud, request, folder, index):
        begin = time.perf_counter()
        def fallback(reason):
            return [],dict(reason=reason,frontend_ms=(time.perf_counter()-begin)*1000)
        if cloud is None:
            return fallback('no_received_cloud')
        age = request['t']-cloud['stamp']
        if not np.isfinite(age) or age < -1e-9 or age > .5:
            return fallback('invalid_cloud_age')
        start = np.asarray(request['start_state']['p'])
        if np.linalg.norm(np.asarray(request['goal'])-start) < 3.:
            return fallback('near_goal_repair')
        try:
            xyz = np.asarray(cloud['points'])
            if xyz.shape != (17*720,3) or not np.isfinite(xyz).all():
                return fallback('unsupported_or_nonfinite_cloud')
            image, state = inputs(cloud,request)
            if not np.isfinite(image).all() or not np.isfinite(state).all():
                return fallback('nonfinite_state')
        except (KeyError, ValueError, TypeError):
            return fallback('invalid_observation')
        prepared = time.perf_counter()
        with self.torch.inference_mode():
            score, logits, delta = self.model(self.torch.from_numpy(image[None]),self.torch.from_numpy(state[None]))
            scores = score.numpy()[0]
            feasible = self.torch.sigmoid(logits).numpy()[0]
            offsets = delta.numpy()[0]
        inferred = time.perf_counter()
        if not all(np.isfinite(a).all() for a in [scores,feasible,offsets]):
            return fallback('nonfinite_prediction')
        routes, utility = decode(request,scores,feasible,offsets,self.fixed)
        record = dict(reason='ok',kind=FORMAT,capture_stamp=cloud['stamp'],capture_age_s=age,
            capture_origin=cloud['origin'],scores=scores.tolist(),predicted_feasibility=feasible.tolist(),
            utility=utility.tolist(),offsets=offsets.tolist(),routes=routes,fixed_geometry=self.fixed,
            observed_fraction=float(image[2].mean()),preprocess_ms=(prepared-begin)*1000,
            inference_ms=(inferred-prepared)*1000)
        folder = Path(folder)
        folder.mkdir(exist_ok=True)
        np.savez_compressed(folder/f'{index:05d}.npz',range_image=image,state=state,
                            scores=scores,feasible=feasible,offsets=offsets)
        save(folder/f'{index:05d}.json',record)
        record['frontend_ms'] = (time.perf_counter()-begin)*1000
        return routes,{k:record[k] for k in ['reason','kind','capture_stamp','capture_age_s',
                       'observed_fraction','preprocess_ms','inference_ms','frontend_ms']}
