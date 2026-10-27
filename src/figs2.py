import sys, json, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
u = pd.read_csv(sys.argv[1], low_memory=False); R = json.load(open(sys.argv[2])); OUT = sys.argv[3]
BLUE, ORANGE, AQUA, INK, MUTED, GRID = '#2a78d6', '#eb6834', '#1baf7a', '#222222', '#6b6b6b', '#e4e4e0'
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'Liberation Serif', 'DejaVu Serif'], 'font.size': 8, 'axes.titlesize': 8,
                     'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7, 'axes.edgecolor': MUTED,
                     'axes.linewidth': 0.6, 'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.grid': True, 'grid.color': GRID,
                     'grid.linewidth': 0.5, 'savefig.dpi': 600, 'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False})
W1, W2 = 3.4, 7.0
def save(fig, name): fig.savefig(f'{OUT}/{name}.png', bbox_inches='tight'); fig.savefig(f'{OUT}/{name}.pdf', bbox_inches='tight'); plt.close(fig)

# Fig 1: RMS ratio vs fraction of life, all evaluable bearings, one axis each (H / x / single)
fig, axs = plt.subplots(1, 3, figsize=(W2, 2.4), sharey=True)
for ax, ds in zip(axs, ['IMS', 'XJTU-SY', 'FEMTO']):
    d = u[(u.dataset == ds) & (u.axis.isin(['H', 'x', 'single'])) & (u.n_snap >= 120)]
    for unit, s in d.groupby('unit'):
        s = s.sort_values('idx'); mu = s.rms[(s.idx > 10) & (s.idx <= 60)].mean()
        r = (s.rms / mu).rolling(5, min_periods=1, center=True).median()
        ax.plot(s.t_h / s.life_h.iloc[0], r, lw=0.6, color=BLUE, alpha=0.55)
    ax.axhline(1.2, color=INK, ls=':', lw=0.7); ax.axhline(2.0, color=INK, ls='--', lw=0.6)
    ax.axvspan(0.85, 1.0, color='#000000', alpha=0.06, lw=0)
    ax.set_yscale('log'); ax.set_yticks([0.5, 1, 2, 5, 10]); ax.set_yticklabels(['0.5', '1', '2', '5', '10']); ax.minorticks_off()
    ax.set_ylim(0.4, 30); ax.set_xlabel('Fraction of life'); ax.set_title(f"{ds} ({d.unit.nunique()} bearings)", loc='left')
axs[0].set_ylabel('RMS / baseline mean (5-pt median)')
save(fig, 'fig1_trajectories')

# Fig 2: operating characteristic over RMS ratio c (pooled, 33 bearings)
cc = pd.DataFrame(R['c_curve'])
fig, ax = plt.subplots(figsize=(W1, 2.4))
ax.plot(cc.c, cc.detected, color=BLUE, marker='o', ms=2.5, lw=1, label='failures detected')
ax.plot(cc.c, cc.transient_units, color=ORANGE, marker='s', ms=2.5, lw=1, label='bearings with transient false alarms')
ax.plot(cc.c, cc.persistent_early_units, color=AQUA, marker='^', ms=2.5, lw=1, label='bearings with persistent early alarm')
ax.axvline(1.2, color=MUTED, ls=':', lw=0.7); ax.axvline(2.0, color=MUTED, ls='--', lw=0.6)
ax.set_xlabel('RMS threshold c (multiple of baseline mean)'); ax.set_ylabel('Number of bearings (of 33)'); ax.set_ylim(0, 34)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=1)
save(fig, 'fig2_operating_characteristic')

# Fig 3: first-alarm timing for the 1.2x rule
t = pd.DataFrame(R['first_alarm_timing']); t = t[t.c == 1.2]
fig, ax = plt.subplots(figsize=(W1, 2.2))
for i, ds in enumerate(['IMS', 'XJTU-SY', 'FEMTO']):
    d = t[t.dataset == ds]; jit = (np.random.default_rng(1).random(len(d)) - 0.5) * 0.35
    pers = d.density_after >= 0.9
    ax.scatter(d.first_life[pers], i + jit[pers.values], s=14, color=BLUE, label='alarm then stays on (>= 90%)' if i == 0 else None, zorder=3)
    ax.scatter(d.first_life[~pers], i + jit[~pers.values], s=14, facecolors='white', edgecolors=ORANGE, linewidths=1.0, label='alarm then mostly off' if i == 0 else None, zorder=3)
ax.axvspan(0.85, 1.0, color='#000000', alpha=0.06, lw=0); ax.text(0.86, 2.45, 'final 15%', fontsize=6.5, color=MUTED)
ax.set_yticks([0, 1, 2]); ax.set_yticklabels(['IMS', 'XJTU-SY', 'FEMTO']); ax.set_ylim(-0.5, 2.7); ax.set_xlim(0, 1.02)
ax.set_xlabel('First 1.2x alarm (fraction of life elapsed)'); ax.grid(axis='y', visible=False)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=2, fontsize=6.5)
save(fig, 'fig3_first_alarm_timing')
print('ok')
