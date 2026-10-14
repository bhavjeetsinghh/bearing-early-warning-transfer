"""Build one long table of per-snapshot features for every bearing axis in IMS, XJTU-SY and FEMTO."""
import sys, numpy as np, pandas as pd
D = sys.argv[1]; OUT = sys.argv[2]
parts = []
# --- IMS (failed bearings only; channel -> axis)
FAILED = {('T1', 3): 'inner race', ('T1', 4): 'rolling element', ('T2', 1): 'outer race', ('T3', 3): 'outer race'}
for run, f in [('T1', 'test1.csv'), ('T2', 'test2.csv'), ('T3', 'test3.csv')]:
    d = pd.read_csv(f'{D}/{f}', parse_dates=['timestamp']).sort_values(['timestamp', 'channel'])
    med = d.groupby('channel').rms.median(); dead = (d.rms < 0.1 * d.channel.map(med)).groupby(d.filename).all()
    bad = []
    for fn in sorted(dead.index, reverse=True):
        if dead[fn]: bad.append(fn)
        else: break
    d = d[~d.filename.isin(bad)]
    ts = pd.Series(sorted(d.timestamp.unique())); dt = ts.diff().dt.total_seconds().fillna(0) / 3600
    clock = pd.DataFrame({'timestamp': ts, 't_h': np.where(dt > 0.25, 10 / 60, dt).cumsum()})
    d = d.merge(clock, on='timestamp')
    for (r, b), mode in FAILED.items():
        if r != run: continue
        for ch in sorted(d[d.bearing == b].channel.unique()):
            s = d[(d.bearing == b) & (d.channel == ch)].sort_values('t_h')
            axis = ('x' if ch % 2 else 'y') if run == 'T1' else 'single'
            parts.append(pd.DataFrame(dict(dataset='IMS', unit=f'{run}-B{b}', mode=mode, condition='2000 rpm / 6000 lb', axis=axis,
                idx=np.arange(1, len(s) + 1), t_h=s.t_h.values, rms=s.rms.values, kurtosis=s.kurtosis_pearson.values,
                skewness=s.skewness.values, peak2peak=s.peak2peak.values)))
# --- XJTU-SY: one file per minute
x = pd.read_csv(f'{D}/xjtu_features.csv')
for (g, b), s in x.groupby(['group', 'bearing']):
    for ax, a in s.groupby('axis'):
        a = a.sort_values('snapshot')
        parts.append(pd.DataFrame(dict(dataset='XJTU-SY', unit=b, mode='', condition=g, axis=ax, idx=a.snapshot.values,
            t_h=(a.snapshot.values - 1) / 60.0, rms=a.rms.values, kurtosis=a.kurtosis_pearson.values, skewness=a.skewness.values, peak2peak=a.peak2peak.values)))
# --- FEMTO: one snapshot every 10 s (snapshot index used as clock)
fm = pd.read_csv(f'{D}/femto_features.csv')
cond = {'1': '1800 rpm / 4000 N', '2': '1650 rpm / 4200 N', '3': '1500 rpm / 5000 N'}
for (g, b), s in fm.groupby(['group', 'bearing']):
    for ax, a in s.groupby('axis'):
        a = a.sort_values('snapshot')
        parts.append(pd.DataFrame(dict(dataset='FEMTO', unit=b, mode='', condition=cond[b[7]], axis=ax, idx=a.snapshot.values,
            t_h=(a.snapshot.values - 1) * 10 / 3600.0, rms=a.rms.values, kurtosis=a.kurtosis_pearson.values, skewness=a.skewness.values, peak2peak=a.peak2peak.values)))
u = pd.concat(parts, ignore_index=True)
u['life_h'] = u.groupby(['dataset', 'unit', 'axis']).t_h.transform('max')
u['n_snap'] = u.groupby(['dataset', 'unit', 'axis']).idx.transform('max')
u['ttg_h'] = u.life_h - u.t_h
u['ttg_frac'] = u.ttg_h / u.life_h
u['crest'] = (u.peak2peak / 2) / u.rms
u.to_csv(OUT, index=False)
s = u.groupby(['dataset', 'unit', 'axis']).agg(n=('idx', 'max'), life_h=('life_h', 'first')).reset_index()
print(s.groupby('dataset').agg(units=('unit', 'nunique'), axes=('axis', 'size')), '\n', s.to_string())
