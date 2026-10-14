"""Post hoc cost check: renewal-reward model of paper 1 (illustrative costs, IMS time scale E[L] = 1,053.2 h)
using transient false-alarm episodes counted per BEARING (all axes summed), not per axis."""
import sys, json, numpy as np
from collections import defaultdict
R = json.load(open(sys.argv[1])); OUT = sys.argv[2]
Cp, Cf, Cfa, EL = 14500.0, 88000.0, 2500.0, 1053.2
age = R['cost']['age_opt']
def cbm(p, lf, f): return 1000 * (p * Cp + (1 - p) * Cf + f * Cfa) / (EL - p * lf * EL)
res = {'age_opt': age, 'rtf': R['cost']['rtf']}
for dn in ['D1_2x', 'D1_1.2x', 'D4']:
    per = defaultdict(int)
    for r in R['bearings']:
        if r['eligible']: per[(r['dataset'], r['unit'])] += r[dn]['transient_episodes']
    v = np.array(list(per.values())); c = R['cost'][dn]; lo = c['ci'][0]
    f_be = {b: (a / 1000 * (EL - lo * c['lead_frac_min'] * EL) - (lo * Cp + (1 - lo) * Cf)) / Cfa for b, a in age.items()}
    res[dn] = dict(p_hat=c['p_hat'], ci_low=lo, lead_frac_min=c['lead_frac_min'], fa_per_bearing_mean=float(v.mean()),
                   fa_per_bearing_median=float(np.median(v)), fa_per_bearing_max=int(v.max()), bearings_with_fa=int((v > 0).sum()),
                   at_phat=cbm(c['p_hat'], c['lead_frac_min'], v.mean()), at_lower=cbm(lo, c['lead_frac_min'], v.mean()),
                   breakeven_fa_at_lower=f_be)
json.dump(res, open(OUT, 'w'), indent=1); print(json.dumps(res, indent=1))
