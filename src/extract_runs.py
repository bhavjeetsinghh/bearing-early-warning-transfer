"""Extract per-snapshot time-domain features from XJTU-SY or FEMTO/PRONOSTIA bearing data.

Usage:
    python extract_runs.py xjtu  <XJTU-SY root folder>  <output.csv>
    python extract_runs.py femto <FEMTO root folder>    <output.csv>

XJTU-SY: <root>/<condition, e.g. 35Hz12kN>/<Bearing1_1>/<1.csv, 2.csv, ...>
         each CSV has a header row and two columns (horizontal, vertical); one file per minute.
FEMTO:   <root>/<Learning_set | Full_Test_Set>/<Bearing1_1>/acc_00001.csv ...
         columns: hour, minute, second, microsecond, horizontal, vertical (no header; ',' or ';').
Output: one row per (bearing, snapshot, axis) with RMS, Pearson kurtosis, skewness, variance, peak-to-peak.
"""
import os, re, sys
import numpy as np, pandas as pd
from scipy.stats import kurtosis, skew

kind, root, out = sys.argv[1], sys.argv[2], sys.argv[3]

def feats(s):
    return dict(rms=float(np.sqrt(np.mean(s ** 2))), kurtosis_pearson=float(kurtosis(s, fisher=False)),
                skewness=float(skew(s)), variance=float(np.var(s)), peak2peak=float(s.max() - s.min()), n_rows=len(s))

rows = []
for dirpath, dirs, files in sorted(os.walk(root, followlinks=True)):
    dirs.sort()
    name = os.path.basename(dirpath)
    if not re.match(r'Bearing\d_\d+$', name):
        continue
    parent = os.path.basename(os.path.dirname(dirpath))
    if kind == 'femto' and parent == 'Test_set':      # truncated copies of Full_Test_Set bearings
        continue
    if kind == 'xjtu':
        snaps = sorted((f for f in files if re.match(r'\d+\.csv$', f)), key=lambda f: int(f[:-4]))
    else:
        snaps = sorted(f for f in files if re.match(r'acc_\d+\.csv$', f))
    if not snaps:
        continue
    print(f'{parent}/{name}: {len(snaps)} snapshots', flush=True)
    for i, f in enumerate(snaps):
        p = os.path.join(dirpath, f)
        if kind == 'xjtu':
            x = pd.read_csv(p).to_numpy(dtype=float)
            hv, t_min = (x[:, 0], x[:, 1]), float(int(f[:-4]))            # file k = minute k
        else:
            x = pd.read_csv(p, header=None, sep=None, engine='python').to_numpy(dtype=float)
            hv = (x[:, 4], x[:, 5])
            t_min = (x[0, 0] * 3600 + x[0, 1] * 60 + x[0, 2] + x[0, 3] * 1e-6) / 60.0   # clock time of the burst
        for axis, s in zip(('H', 'V'), hv):
            rows.append(dict(dataset=kind, group=parent, bearing=name, snapshot=i + 1, file=f, axis=axis, t_min_raw=t_min, **feats(s)))
df = pd.DataFrame(rows)
if df.empty:
    sys.exit('No bearing folders found under ' + root + ' (expected folders named like Bearing1_1).')
if kind == 'femto':   # clock time wraps at midnight: unwrap per bearing
    for (g, b), d in df.groupby(['group', 'bearing']):
        t = d.drop_duplicates('snapshot').set_index('snapshot').t_min_raw
        add = (t.diff() < -600).cumsum() * 1440
        df.loc[d.index, 't_min_raw'] = d.snapshot.map(t + add)
df.to_csv(out, index=False)
print(f'Wrote {len(df)} rows for {df.bearing.nunique()} bearings to {out}')
