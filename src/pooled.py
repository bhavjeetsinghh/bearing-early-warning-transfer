"""Paper 2: pooled label-free evaluation across IMS, XJTU-SY and FEMTO (protocol in PROTOCOL.md).
Usage: python pooled.py out/pooled.csv out
"""
import sys, json, numpy as np, pandas as pd
from scipy import stats

u = pd.read_csv(sys.argv[1], low_memory=False); OUT = sys.argv[2]
RUNIN, BASE_END, MIN_SNAP = 10, 60, 120

def persist(f, k):
    f = np.asarray(f, bool); c = 0; o = np.zeros(len(f), bool)
    for i, x in enumerate(f):
        c = c + 1 if x else 0; o[i] = c >= k
    return o

def cp_ci(k, n):
    lo = stats.beta.ppf(0.025, k, n - k + 1) if k > 0 else 0.0
    hi = stats.beta.ppf(0.975, k + 1, n - k) if k < n else 1.0
    return float(lo), float(hi)

def scores(s):
    """detector scores and thresholds for one bearing-axis series s (sorted by idx)"""
    base = ((s.idx > RUNIN) & (s.idx <= BASE_END)).values
    mu, sd = s.rms[base].mean(), s.rms[base].std()
    X = np.c_[np.log(s.rms), s['kurtosis'], s['skewness'], s.crest]
    Xb = X[base]; m0 = Xb.mean(0); Ci = np.linalg.pinv(np.cov(Xb, rowvar=False)); n, p = Xb.shape
    T2 = np.einsum('ij,jk,ik->i', X - m0, Ci, X - m0)
    ucl = p * (n + 1) * (n - 1) / (n * (n - p)) * stats.f.ppf(1 - 1e-4, p, n - p)
    return mu, {'D1_2x': (s.rms.values, 2 * mu), 'D1_1.2x': (s.rms.values, 1.2 * mu), 'D2': (s.rms.values, mu + 3 * sd),
                'D3': (s['kurtosis'].values, 6.0), 'D4': (T2, ucl)}

def evaluate(s, score, thr, k=3, final=0.15):
    mon = (s.idx > BASE_END).values
    al = persist(score > thr, k) & mon
    ref = (mon & (s.ttg_frac > final)).values
    rising = al & ~np.r_[False, al[:-1]]
    ep = int((rising & ref).sum()); nref = int(ref.sum())
    idx = np.where(al)[0]; sus = None
    life = s.life_h.iloc[0]
    if len(idx) and idx[-1] >= len(s) - 3:
        j = len(idx) - 1
        while j > 0 and (s.t_h.iloc[idx[j]] - s.t_h.iloc[idx[j - 1]]) <= 0.01 * life + 1e-12: j -= 1
        sus = float(s.ttg_h.iloc[idx[j]])
    # post hoc split (not part of the fixed protocol): an episode that starts inside the healthy-reference zone and
    # continues as the sustained block to the end is 'persistent'; all other reference-zone episodes are 'transient'
    persistent = 0
    if sus is not None and ref[idx[j]] and rising[idx[j]]:
        persistent = 1
    return dict(fa_episodes=ep, ref_snapshots=nref, sustained_h=sus, sustained_frac=(sus / life) if sus is not None else None,
                transient_episodes=ep - persistent, persistent_early=persistent)

def run(c_grid=None, k=3, final=0.15, min_frac=0.01, axes=None):
    rows = []
    for (ds, unit, ax), s in u.groupby(['dataset', 'unit', 'axis'], sort=False):
        if axes and ax not in axes and ax != 'single' and not (ds == 'IMS' and ax == 'x'): continue
        s = s.sort_values('idx').reset_index(drop=True)
        eligible = s.idx.max() >= MIN_SNAP
        row = dict(dataset=ds, unit=unit, axis=ax, condition=s.condition.iloc[0], n=int(s.idx.max()), life_h=float(s.life_h.iloc[0]), eligible=bool(eligible))
        if eligible:
            mu, sc = scores(s); row['base_rms'] = float(mu)
            for dn, (score, thr) in sc.items():
                row[dn] = evaluate(s, score, thr, k, final)
            if c_grid is not None:
                for c in c_grid:
                    row[f'c{c:.2f}'] = evaluate(s, s.rms.values, c * mu, k, final)
        rows.append(row)
    return rows

def summarize(rows, dn, min_frac=0.01, datasets=None):
    units = {}
    for r in rows:
        if not r['eligible'] or (datasets and r['dataset'] not in datasets): continue
        key = (r['dataset'], r['unit'])
        e = r[dn]; det = e['sustained_frac'] is not None and e['sustained_frac'] >= min_frac
        u_ = units.setdefault(key, dict(detected=False, lead_h=None, lead_frac=None, fa_axes=0, axes=0, fa_ep=0, ref=0, tr_axes=0, tr_ep=0, pers_axes=0))
        u_['axes'] += 1; u_['fa_axes'] += int(e['fa_episodes'] > 0); u_['fa_ep'] += e['fa_episodes']; u_['ref'] += e['ref_snapshots']
        u_['tr_axes'] += int(e['transient_episodes'] > 0); u_['tr_ep'] += e['transient_episodes']; u_['pers_axes'] += e['persistent_early']
        if det and (u_['lead_frac'] is None or e['sustained_frac'] > u_['lead_frac']):
            u_['detected'] = True; u_['lead_h'] = e['sustained_h']; u_['lead_frac'] = e['sustained_frac']
    n = len(units); k = sum(v['detected'] for v in units.values())
    fa_units = sum(v['fa_axes'] > 0 for v in units.values())
    leads = [v['lead_frac'] for v in units.values() if v['detected']]
    lead_h = [v['lead_h'] for v in units.values() if v['detected']]
    return dict(n=n, detected=k, p_hat=k / n if n else None, ci95=cp_ci(k, n) if n else None,
                fa_units=fa_units, fa_axes=sum(v['fa_axes'] for v in units.values()), axes=sum(v['axes'] for v in units.values()),
                fa_per_1000_snap=1000 * sum(v['fa_ep'] for v in units.values()) / max(1, sum(v['ref'] for v in units.values())),
                transient_units=sum(v['tr_axes'] > 0 for v in units.values()), transient_axes=sum(v['tr_axes'] for v in units.values()),
                transient_per_1000_snap=1000 * sum(v['tr_ep'] for v in units.values()) / max(1, sum(v['ref'] for v in units.values())),
                persistent_early_units=sum(v['pers_axes'] > 0 for v in units.values()),
                median_lead_frac=float(np.median(leads)) if leads else None, median_lead_h=float(np.median(lead_h)) if lead_h else None,
                units={f'{a}|{b}': v for (a, b), v in units.items()})

DET = ['D1_2x', 'D1_1.2x', 'D2', 'D3', 'D4']
grid = [round(x, 2) for x in np.arange(1.10, 3.001, 0.05)]
main = run(c_grid=grid)
R = dict(protocol=dict(runin=RUNIN, base_end=BASE_END, min_snap=MIN_SNAP, final=0.15, k=3, min_lead_frac=0.01))
R['bearings'] = [{k: v for k, v in r.items() if not k.startswith('c')} for r in main]
R['excluded'] = sorted({(r['dataset'], r['unit'], r['n']) for r in main if not r['eligible']})
R['pooled'] = {dn: summarize(main, dn) for dn in DET}
R['by_dataset'] = {dn: {ds: summarize(main, dn, datasets=[ds]) for ds in ['IMS', 'XJTU-SY', 'FEMTO']} for dn in DET}
# operating characteristic over c (pooled)
R['c_curve'] = [dict(c=c, **{k: v for k, v in summarize(main, f'c{c:.2f}').items() if k != 'units'}) for c in grid]
# leave-one-dataset-out transfer
tr = {}
for hold in ['IMS', 'XJTU-SY', 'FEMTO']:
    train = [d for d in ['IMS', 'XJTU-SY', 'FEMTO'] if d != hold]
    chosen = None
    for c in grid:
        s = summarize(main, f'c{c:.2f}', datasets=train)
        if s['fa_axes'] / s['axes'] <= 0.05: chosen = c; break
    if chosen is None: chosen = grid[-1]
    st = summarize(main, f'c{chosen:.2f}', datasets=[hold]); s_tr = summarize(main, f'c{chosen:.2f}', datasets=train)
    tr[hold] = dict(c=chosen, train_fa_axes=s_tr['fa_axes'], train_axes=s_tr['axes'], train_detected=s_tr['detected'], train_n=s_tr['n'],
                    test_detected=st['detected'], test_n=st['n'], test_ci=st['ci95'], test_fa_units=st['fa_units'], test_median_lead_frac=st['median_lead_frac'])
R['transfer'] = tr
# sensitivity
sens = {}
for label, kw, mf in [('final10', dict(final=0.10), 0.01), ('final25', dict(final=0.25), 0.01), ('k1', dict(k=1), 0.01), ('k5', dict(k=5), 0.01),
                      ('lead0.5pct', {}, 0.005), ('lead2pct', {}, 0.02), ('horizontal_only', dict(axes=['H']), 0.01)]:
    rr = run(**kw)
    sens[label] = {dn: {k: v for k, v in summarize(rr, dn, min_frac=mf).items() if k != 'units'} for dn in ['D1_2x', 'D1_1.2x', 'D4']}
R['sensitivity'] = sens
json.dump(R, open(f'{OUT}/pooled.json', 'w'), indent=1, default=str)

f2 = lambda v: '-' if v is None else f'{v:.2f}'
print('excluded:', R['excluded'])
for dn in DET:
    p = R['pooled'][dn]
    print(f"{dn:8s} detected {p['detected']}/{p['n']} CI {p['ci95'][0]:.2f}-{p['ci95'][1]:.2f} | transient-FA bearings {p['transient_units']} ({p['transient_per_1000_snap']:.2f}/1000) persistent-early {p['persistent_early_units']} | FA bearings {p['fa_units']}/{p['n']} axes {p['fa_axes']}/{p['axes']} rate {p['fa_per_1000_snap']:.2f}/1000 snap | median lead {f2(p['median_lead_frac'])} of life ({f2(p['median_lead_h'])} h) | " +
          ' '.join(f"{ds}:{R['by_dataset'][dn][ds]['detected']}/{R['by_dataset'][dn][ds]['n']}(FA{R['by_dataset'][dn][ds]['fa_units']})" for ds in ['IMS', 'XJTU-SY', 'FEMTO']))
print('transfer', json.dumps(tr))
for c in R['c_curve'][::4]: print('c', c['c'], c['detected'], '/', c['n'], 'FA bearings', c['fa_units'], 'FA axes', c['fa_axes'], '/', c['axes'])
for lab, v in sens.items(): print(lab, {dn: (x['detected'], x['n'], x['fa_units'], x['transient_units']) for dn, x in v.items()})

# ---------------- post hoc analyses (labelled as such in the paper) ----------------
# (1) leave-one-dataset-out transfer using the transient false-alarm definition
tr2 = {}
for hold in ['IMS', 'XJTU-SY', 'FEMTO']:
    train = [d for d in ['IMS', 'XJTU-SY', 'FEMTO'] if d != hold]
    chosen = None
    for c in grid:
        s = summarize(main, f'c{c:.2f}', datasets=train)
        if s['transient_axes'] / s['axes'] <= 0.05: chosen = c; break
    if chosen is None: chosen = grid[-1]
    st = summarize(main, f'c{chosen:.2f}', datasets=[hold]); s_tr = summarize(main, f'c{chosen:.2f}', datasets=train)
    tr2[hold] = dict(c=chosen, met=(s_tr['transient_axes'] / s_tr['axes'] <= 0.05), train_transient_axes=s_tr['transient_axes'], train_axes=s_tr['axes'],
                     test_detected=st['detected'], test_n=st['n'], test_ci=st['ci95'], test_transient_units=st['transient_units'])
R['transfer_transient'] = tr2
# (2) per-axis transient false-alarm episodes per bearing life (scale-free input for the cost model)
fa_life = {}
for dn in DET + ['c1.50']:
    eps = [r[dn]['transient_episodes'] for r in main if r['eligible']]
    fa_life[dn] = dict(mean=float(np.mean(eps)), axes=len(eps), share_axes_with_fa=float(np.mean([e > 0 for e in eps])))
R['transient_per_life'] = fa_life
# (3) first-alarm timing for the 1.2x and 2x rules (fraction of life elapsed)
timing = []
for (ds, unit, ax), s in u.groupby(['dataset', 'unit', 'axis'], sort=False):
    s = s.sort_values('idx').reset_index(drop=True)
    if s.idx.max() < MIN_SNAP: continue
    base = ((s.idx > RUNIN) & (s.idx <= BASE_END)).values; mu = s.rms[base].mean()
    for c in (1.2, 2.0):
        al = persist(s.rms.values > c * mu, 3) & (s.idx > BASE_END).values
        i = np.where(al)[0]
        if len(i):
            timing.append(dict(dataset=ds, unit=unit, axis=ax, c=c, first_life=float(s.t_h.iloc[i[0]] / s.life_h.iloc[0]), density_after=float(al[i[0]:].mean())))
R['first_alarm_timing'] = timing
# (4) cost: paper-1 renewal-reward model, IMS time scale, transient false alarms per life as lambda * E[L]
from math import gamma
from scipy import integrate, optimize
Cp, Cf, Cfa, EL = 14500.0, 88000.0, 2500.0, 1053.2
def age_opt(b):
    eta = EL / gamma(1 + 1 / b)
    f = lambda T: (Cp * np.exp(-(T / eta) ** b) + Cf * (1 - np.exp(-(T / eta) ** b))) / integrate.quad(lambda t: np.exp(-(t / eta) ** b), 0, T)[0]
    return optimize.minimize_scalar(f, bounds=(1, 5 * eta), method='bounded').fun
def cbm(pd_, lead_frac, fa_per_life):
    ELc = EL - pd_ * lead_frac * EL
    return (pd_ * Cp + (1 - pd_) * Cf + fa_per_life * Cfa) / ELc
cost = {'age_opt': {str(b): 1000 * age_opt(b) for b in (1.5, 2.0, 3.0)}, 'rtf': 1000 * Cf / EL}
for dn in ['D1_2x', 'D1_1.2x', 'D4']:
    p = R['pooled'][dn]
    leads = [v['lead_frac'] for v in p['units'].values() if v['detected']]
    lf = float(np.min(leads)); fpl = fa_life[dn]['mean']
    cost[dn] = dict(p_hat=p['p_hat'], ci=p['ci95'], lead_frac_min=lf, fa_per_life=fpl,
                    at_phat=1000 * cbm(p['p_hat'], lf, fpl), at_lower=1000 * cbm(p['ci95'][0], lf, fpl),
                    max_fa_per_life_vs_age2=float((age_opt(2.0) * (EL - p['ci95'][0] * lf * EL) - (p['ci95'][0] * Cp + (1 - p['ci95'][0]) * Cf)) / Cfa))
R['cost'] = cost
json.dump(R, open(f'{OUT}/pooled.json', 'w'), indent=1, default=str)
print('transfer_transient', json.dumps(tr2)); print('fa per life', json.dumps(fa_life)); print('cost', json.dumps(cost, indent=0))
