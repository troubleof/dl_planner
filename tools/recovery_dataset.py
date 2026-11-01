"""Portable access to retained training arrays and observed failure events."""
import json
from pathlib import Path
import numpy as np
from recovery_frontend import HORIZONS


def load_dataset(folder, base_only=False):
    folder = Path(folder).resolve()
    with np.load(folder/'dataset.npz') as loaded:
        data = {key: loaded[key] for key in loaded.files}
    rows = json.loads((folder/'records.json').read_text())
    protocol = json.loads((folder/'protocol.json').read_text())
    if 'label_events' in protocol:
        events_path = folder/protocol['label_events']
    else:
        # Backward compatibility for the original, unmodified local datasets.
        parent = Path(protocol.get('parent_dataset', folder))
        if not parent.is_absolute(): parent = folder/parent
        events_path = parent/'future_invalidations.json'
    events = json.loads(events_path.read_text())
    if base_only:
        count = int(protocol['base_states'])
        if not protocol.get('base_prefix_arrays_verified'):
            raise ValueError('The base prefix must be verified before using --base-only')
        data = {key: value[:count] for key, value in data.items()}
        rows = rows[:count]
    assert all(len(value) == len(rows) for value in data.values())
    corrected = 0
    for event in events:
        i, j = event['state'], event['candidate']
        if i >= len(rows): continue
        for h, delta in enumerate(HORIZONS, 1):
            if event['observed_at'] <= rows[i]['query']['t'] + delta + 1e-9:
                corrected += int(data['mask'][i, j, h] == 0)
                data['labels'][i, j, h] = 0
                data['mask'][i, j, h] = 1
    return data, rows, protocol, corrected
