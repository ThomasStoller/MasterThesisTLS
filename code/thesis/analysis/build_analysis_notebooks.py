"""Generates the analysis notebooks 01-05 in code/thesis/analysis.

All of them read only the artifacts that the modelling notebooks export
(code/thesis/modeling/artifacts/<experiment>/), see analysis_utils.py.
Rerun this script only if you want to regenerate the notebooks from scratch
(it overwrites them, including any edits and outputs).
"""
import pathlib

import nbformat as nbf

HERE = pathlib.Path(__file__).resolve().parent

SETUP = """import sys, pathlib, importlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sys.path.append(str(pathlib.Path.cwd()))  # the notebooks live in code/thesis/analysis
import analysis_utils as au
importlib.reload(au)

RESULTS = pathlib.Path('results')
FIGURES = RESULTS / 'figures'
FIGURES.mkdir(parents=True, exist_ok=True)

EXPS = au.available_experiments()
print('experiments with exported artifacts:', EXPS)
missing = [e for e in au.EXPERIMENTS if e not in EXPS]
if missing:
    print('NOT YET RERUN (no artifacts):', missing)"""


def md(t):
    return nbf.v4.new_markdown_cell(t.strip("\n"))


def code(t):
    return nbf.v4.new_code_cell(t.strip("\n"))


def write(name, cells):
    nb = nbf.v4.new_notebook()
    nb["metadata"]["kernelspec"] = {"display_name": "masterthesis", "language": "python", "name": "masterthesis"}
    nb["cells"] = cells
    nbf.write(nb, HERE / name)
    print("wrote", name)


# ======================================================================================
# 01 metrics and significance
# ======================================================================================
write("01_metrics_and_significance.ipynb", [
    md("""
# 01 Metrics, binary task and significance tests

Reads the predictions that the modelling notebooks exported and produces:

1. one table with every metric of every run (macro F1, MCC, AUC-PR, balanced accuracy,
   per class F1, and the per source majority vote),
2. the binary against three class comparison for **H2**,
3. confidence intervals with a **bootstrap over source addresses**,
4. significance tests for **H1** (distance), **H3** (packet against grouped split),
   **H4** (in domain against transfer) and the model comparison,
5. optionally the variability over the two hour blocks (leave one block out, retrains the
   random forest).

Why a bootstrap over sources and not over packets: the packets of one address are strongly
correlated, so treating them as independent samples gives intervals that are far too narrow
and p values that are far too small. Resampling whole addresses respects that structure.
McNemar is reported as well, because it is the textbook paired test, but it treats packets as
independent and is therefore optimistic.

Sections 1 to 4 need no training and take seconds to a few minutes.
"""),
    code(SETUP),
    md("## 1. All metrics of all runs"),
    code("""
m = au.load_all_metrics()
m['exp'] = m['experiment'].map(au.exp_name)
cols = ['exp', 'label', 'model', 'split', 'frame', 'macro_f1', 'mcc', 'auc_pr_macro', 'auc_pr_targets',
        'balanced_acc', 'acc', 'f1_0', 'f1_1', 'f1_2',
        'src_n', 'src_macro_f1', 'src_mcc', 'src_target_recall']
table = m[[c for c in cols if c in m.columns]].sort_values(['label', 'frame', 'model', 'split', 'exp'])
table.to_csv(RESULTS / 'metrics_all_runs.csv', index=False)
print('wrote', RESULTS / 'metrics_all_runs.csv')
table[table['model'] == 'Random Forest'].round(3)
"""),
    md("""
### Random forest overview: packet level against per source majority vote

`src_*` columns: one decision per address (majority of its packet predictions). This is the
deployment view from the design protocol table. `src_target_recall` is the share of group
addresses that are assigned to the correct group.
"""),
    code("""
rf = table[table['model'] == 'Random Forest']
overview = rf.pivot_table(index=['label', 'exp'], columns=['frame', 'split'],
                          values=['macro_f1', 'mcc', 'auc_pr_macro', 'src_macro_f1'])
overview.round(3)
"""),
    md("""
## 2. Binary against three class task (H2)

H2 expects that **detection** (background against any group) stays strong while the three class
task mostly fails at **attribution** (group 1 against group 2).

Two views:
* the binary models themselves (label `binary`), and
* the three class random forest read as a detector: of all target packets, how many are
  detected as *some* group, how many get the correct group, how many are confused with the
  other group and how many are lost to the background.
"""),
    code("""
rows = []
for e in EXPS:
    for split in ['packet', 'grouped']:
        for frame in ['indomain', 'transfer']:
            p = au.load_predictions(e, 'multi', 'random_forest', split, frame, with_meta=False)
            t, y = p['y_true'].to_numpy(), p['y_pred'].to_numpy()
            tgt = t != 0
            n = max(tgt.sum(), 1)
            err = tgt & (y != t)
            confused = tgt & (y != 0) & (y != t)
            rows.append(dict(exp=au.exp_name(e), split=split, frame=frame,
                             target_packets=int(tgt.sum()),
                             detected_as_any_group=(tgt & (y != 0)).sum() / n,
                             correct_group=(tgt & (y == t)).sum() / n,
                             confused_with_other_group=confused.sum() / n,
                             lost_to_background=(tgt & (y == 0)).sum() / n,
                             group_confusion_share_of_errors=confused.sum() / max(err.sum(), 1)))
h2 = pd.DataFrame(rows)

binary = (m[(m['model'] == 'Random Forest') & (m['label'] == 'binary')]
          .set_index(['exp', 'split', 'frame'])[['macro_f1', 'mcc', 'f1_1']]
          .rename(columns={'macro_f1': 'binary_macro_f1', 'mcc': 'binary_mcc', 'f1_1': 'binary_f1_target'}))
multi = (m[(m['model'] == 'Random Forest') & (m['label'] == 'multi')]
         .set_index(['exp', 'split', 'frame'])[['macro_f1', 'mcc']]
         .rename(columns={'macro_f1': 'multi_macro_f1', 'mcc': 'multi_mcc'}))
h2 = h2.join(binary, on=['exp', 'split', 'frame']).join(multi, on=['exp', 'split', 'frame'])
h2.to_csv(RESULTS / 'h2_binary_vs_multi.csv', index=False)
h2.round(3)
"""),
    code("""
sub = h2[(h2['split'] == 'grouped') & (h2['frame'] == 'transfer')].set_index('exp')
ax = sub[['correct_group', 'confused_with_other_group', 'lost_to_background']].plot.bar(
    stacked=True, figsize=(9, 4), color=['tab:green', 'tab:orange', 'tab:grey'])
ax.set_ylabel('share of target packets')
ax.set_title('Three class random forest on the university capture (grouped split)')
ax.legend(loc='upper right', fontsize=8)
plt.xticks(rotation=15, ha='right'); plt.tight_layout()
plt.savefig(FIGURES / 'fig_h2_target_packet_outcome.pdf'); plt.show()
"""),
    md("""
## 3. Confidence intervals (bootstrap over source addresses)

`MODELS` controls which learners are included. On the university capture all runs of one
experiment and label share the same packets, so they are bootstrapped together (paired). The
in domain test sets differ between the packet and the grouped split, so they are bootstrapped
separately.
"""),
    code("""
N_BOOT = 1000
LABELS = ['multi', 'binary']
MODELS = ['random_forest', 'multi_layer_perceptron', 'stochastic_gradient_descent']
SPLITS = ['packet', 'grouped']
METRICS = ['macro_f1', 'mcc']

boot = {}      # (exp, label, frame, metric) -> DataFrame with one column per 'model|split'
ci_rows = []
for ei, e in enumerate(EXPS):
    for label in LABELS:
        # transfer: paired over all runs
        preds = {f'{mo}|{sp}': au.load_predictions(e, label, mo, sp, 'transfer') for mo in MODELS for sp in SPLITS}
        first = next(iter(preds.values()))
        for metric in METRICS:
            b = au.cluster_bootstrap(first['source'], first['y_true'], {k: v['y_pred'] for k, v in preds.items()},
                                     metric, N_BOOT, seed=100 + ei)
            boot[(e, label, 'transfer', metric)] = b
        # in domain: one bootstrap per run
        for mo in MODELS:
            for sp in SPLITS:
                p = au.load_predictions(e, label, mo, sp, 'indomain')
                for metric in METRICS:
                    b = au.cluster_bootstrap(p['source'], p['y_true'], {f'{mo}|{sp}': p['y_pred']},
                                             metric, N_BOOT, seed=200 + ei)
                    boot.setdefault((e, label, 'indomain', metric), pd.DataFrame({'replicate': b['replicate']}))
                    boot[(e, label, 'indomain', metric)][f'{mo}|{sp}'] = b[f'{mo}|{sp}']

for (e, label, frame, metric), b in boot.items():
    for run in [c for c in b.columns if c != 'replicate']:
        mo, sp = run.split('|')
        lo, hi = au.ci(b[run])
        ci_rows.append(dict(exp=au.exp_name(e), label=label, frame=frame, metric=metric, model=mo, split=sp,
                            boot_mean=b[run].mean(), ci_low=lo, ci_high=hi))
cis = pd.DataFrame(ci_rows)
point = (m.assign(model=m['model'].map(au.slug))
          .melt(id_vars=['exp', 'label', 'model', 'split', 'frame'], value_vars=METRICS,
                var_name='metric', value_name='point'))
cis = cis.merge(point, on=['exp', 'label', 'model', 'split', 'frame', 'metric'], how='left')
cis.to_csv(RESULTS / 'confidence_intervals.csv', index=False)
cis[(cis['model'] == 'random_forest') & (cis['metric'] == 'macro_f1')].round(3)
"""),
    code("""
# Transfer macro F1 of the random forest with 95 % intervals
sub = cis[(cis['model'] == 'random_forest') & (cis['metric'] == 'macro_f1') & (cis['frame'] == 'transfer')]
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, label in zip(axes, LABELS):
    s = sub[sub['label'] == label]
    for k, sp in enumerate(SPLITS):
        q = s[s['split'] == sp]
        x = np.arange(len(q)) + (k - 0.5) * 0.3
        ax.errorbar(x, q['point'], yerr=[q['point'] - q['ci_low'], q['ci_high'] - q['point']],
                    fmt='o', capsize=3, label=sp)
    ax.set_xticks(np.arange(len(q))); ax.set_xticklabels(q['exp'], rotation=20, ha='right')
    ax.axhline(1 / 3 if label == 'multi' else 0.5, ls='--', lw=1, color='grey')
    ax.set_title(f'{label} task, university capture'); ax.set_ylim(0, 1)
axes[0].set_ylabel('macro F1 (95 % source bootstrap)'); axes[0].legend()
plt.tight_layout(); plt.savefig(FIGURES / 'fig_ci_transfer_macro_f1.pdf'); plt.show()
"""),
    md("""
## 4. Significance tests

* **H3** packet against grouped split: on the university capture the two runs are paired (same
  packets): paired source bootstrap and McNemar. In domain the test sets differ: unpaired.
* **H4** in domain against transfer: unpaired (different test sets).
* **H1** same against different distance (E1/E2, E3/E4, E5/E6): unpaired.
* **Models**: random forest against MLP and SGD on the university capture: paired.

All p values of one hypothesis family are Holm corrected. Primary metric: macro F1.
"""),
    code("""
METRIC = 'macro_f1'
tests = []

def add(hyp, exp, label, frame, a, b, res):
    tests.append(dict(hypothesis=hyp, exp=exp, label=label, frame=frame, a=a, b=b, **res))

for e in EXPS:
    for label in LABELS:
        bt = boot[(e, label, 'transfer', METRIC)]
        bi = boot[(e, label, 'indomain', METRIC)]
        # H3, transfer, paired
        res = au.paired_bootstrap_test(bt, 'random_forest|packet', 'random_forest|grouped')
        pa = au.load_predictions(e, label, 'random_forest', 'packet', 'transfer', with_meta=False)
        pg = au.load_predictions(e, label, 'random_forest', 'grouped', 'transfer', with_meta=False)
        res.update(au.mcnemar(pa['y_true'], pa['y_pred'], pg['y_pred']))
        add('H3', au.exp_name(e), label, 'transfer', 'packet', 'grouped', res)
        # H3, in domain, unpaired
        add('H3', au.exp_name(e), label, 'indomain', 'packet', 'grouped',
            au.unpaired_bootstrap_test(bi['random_forest|packet'], bi['random_forest|grouped']))
        # H4, grouped split
        add('H4', au.exp_name(e), label, 'indomain-transfer', 'indomain', 'transfer',
            au.unpaired_bootstrap_test(bi['random_forest|grouped'], bt['random_forest|grouped']))
        # models, transfer, paired
        for other in [mo for mo in MODELS if mo != 'random_forest']:
            for sp in SPLITS:
                res = au.paired_bootstrap_test(bt, f'random_forest|{sp}', f'{other}|{sp}')
                add('models', au.exp_name(e), label, f'transfer/{sp}', 'random_forest', other, res)

# H1: same against different distance
for same, diff in au.DISTANCE_PAIRS:
    if same not in EXPS or diff not in EXPS:
        continue
    for label in LABELS:
        for frame in ['transfer', 'indomain']:
            for sp in SPLITS:
                add('H1', f'{au.exp_name(same)} / {au.exp_name(diff)}', label, f'{frame}/{sp}', 'same', 'diff',
                    au.unpaired_bootstrap_test(boot[(same, label, frame, METRIC)][f'random_forest|{sp}'],
                                               boot[(diff, label, frame, METRIC)][f'random_forest|{sp}']))

tests = pd.DataFrame(tests)
tests['p_holm'] = np.nan
for hyp, idx in tests.groupby('hypothesis').groups.items():
    tests.loc[idx, 'p_holm'] = au.holm(tests.loc[idx, 'p_boot'])
tests['significant_5pct'] = tests['p_holm'] < 0.05
tests.to_csv(RESULTS / 'significance_tests.csv', index=False)
tests.round(4)
"""),
    md("""
## 5. Variability over the two hour blocks (optional, retrains)

Leave one block out with the random forest: train on all but one two hour block, test on the
left out block (a later or earlier session of the same campaign) and on the university
capture. This gives one score per block, so the spread between blocks can be reported (the
design argues for five independent recording units).

Runtime: one random forest fit per block, label and experiment (about 1 to 3 minutes each).
"""),
    code("""
RUN_BLOCK_CV = True
BLOCK_LABELS = ['group_label_multi', 'group_label']

if RUN_BLOCK_CV:
    from sklearn.metrics import f1_score, matthews_corrcoef
    block_rows = []
    for e in EXPS:
        tr = au.load_frame(e, 'train')
        ev = au.load_frame(e, 'eval')
        feats = au.feature_columns(tr)
        for label in BLOCK_LABELS:
            for b in sorted(tr['block'].unique()):
                held = tr['block'] == b
                model = au.make_random_forest().fit(tr.loc[~held, feats], tr.loc[~held, label])
                for frame, df in [('left out block', tr[held]), ('transfer', ev)]:
                    y = model.predict(df[feats])
                    block_rows.append(dict(exp=au.exp_name(e), label=label, block=int(b), frame=frame,
                                           macro_f1=f1_score(df[label], y, average='macro', zero_division=0),
                                           mcc=matthews_corrcoef(df[label], y)))
                print(e, label, 'block', b, 'done')
            del model
    blocks = pd.DataFrame(block_rows)
    blocks.to_csv(RESULTS / 'block_variability.csv', index=False)
    display(blocks.groupby(['label', 'frame', 'exp'])[['macro_f1', 'mcc']].agg(['mean', 'std', 'min', 'max']).round(3))
"""),
])

# ======================================================================================
# 02 feature family ablation
# ======================================================================================
write("02_feature_family_ablation.ipynb", [
    md("""
# 02 Feature family ablation (H6)

For every experiment the random forest of the modelling notebooks (same hyperparameters, same
grouped split with seed 1) is retrained

* with the full set of 42 features,
* with one of the eight feature families removed (`drop`),
* optionally with only one family (`only`, set `RUN_ONLY = True`).

Every configuration is evaluated in domain (held out sources of the training capture) and on
the university capture. The difference to the full set on the university capture is tested with
a paired bootstrap over source addresses, Holm corrected over the eight families per experiment.

H6 is supported if every `drop` configuration scores below the full set; it is refuted if at
least one `drop` configuration matches or exceeds it.

Runtime: (1 + 8 [+ 8]) random forest fits per experiment and label, about 1 to 3 minutes each.
Results are cached per experiment in `results/ablation/`, so an interrupted run can continue.
"""),
    code(SETUP),
    code("""
from sklearn.metrics import f1_score, matthews_corrcoef

LABELS = ['group_label_multi']        # add 'group_label' to repeat it on the binary task
SPLIT_GROUPED = True                  # the honest in domain estimate, as in the design
RUN_ONLY = False                      # also train with a single family
N_BOOT = 1000
OUT = RESULTS / 'ablation'
OUT.mkdir(exist_ok=True)

def configurations(feats):
    yield 'full', feats
    for fam, cols in au.FEATURE_FAMILIES.items():
        yield f'drop: {fam}', [f for f in feats if f not in cols]
    if RUN_ONLY:
        for fam, cols in au.FEATURE_FAMILIES.items():
            yield f'only: {fam}', [f for f in feats if f in cols]
"""),
    code("""
for e in EXPS:
    for label in LABELS:
        cache = OUT / f'{e}__{label}.parquet'
        if cache.exists():
            print('cached:', cache.name)
            continue
        tr = au.load_frame(e, 'train')
        ev = au.load_frame(e, 'eval')
        feats = au.feature_columns(tr)
        tr_idx, te_idx = au.grouped_split(tr, tr[label], grouped=SPLIT_GROUPED)
        fit_df, test_df = tr.iloc[tr_idx], tr.iloc[te_idx]
        transfer = pd.DataFrame({'y_true': ev[label].to_numpy(), 'source': ev['source'].to_numpy()})
        indomain = pd.DataFrame({'y_true': test_df[label].to_numpy(), 'source': test_df['source'].to_numpy()})
        for name, cols in configurations(feats):
            model = au.make_random_forest().fit(fit_df[cols], fit_df[label])
            indomain[name] = model.predict(test_df[cols])
            transfer[name] = model.predict(ev[cols])
            print(f'{e} {label} {name:35s} in domain {f1_score(indomain.y_true, indomain[name], average="macro"):.3f}'
                  f'  transfer {f1_score(transfer.y_true, transfer[name], average="macro"):.3f}')
            del model
        indomain.assign(frame='indomain').pipe(lambda d: pd.concat([d, transfer.assign(frame='transfer')])) \\
            .to_parquet(cache, index=False)
"""),
    md("## Scores and paired tests against the full set"),
    code("""
rows = []
for e in EXPS:
    for label in LABELS:
        d = pd.read_parquet(OUT / f'{e}__{label}.parquet')
        configs = [c for c in d.columns if c not in ('y_true', 'source', 'frame')]
        for frame, f in d.groupby('frame'):
            b = au.cluster_bootstrap(f['source'], f['y_true'], {c: f[c] for c in configs}, 'macro_f1', N_BOOT)
            for c in configs:
                r = dict(exp=au.exp_name(e), label=label, frame=frame, config=c,
                         macro_f1=f1_score(f['y_true'], f[c], average='macro', zero_division=0),
                         mcc=matthews_corrcoef(f['y_true'], f[c]),
                         **{f'f1_{k}': v for k, v in enumerate(f1_score(f['y_true'], f[c], average=None, zero_division=0))})
                if c != 'full':
                    r.update(au.paired_bootstrap_test(b, c, 'full'))
                rows.append(r)
abl = pd.DataFrame(rows)
abl['delta_vs_full'] = abl['macro_f1'] - abl.groupby(['exp', 'label', 'frame'])['macro_f1'].transform(
    lambda s: s[abl.loc[s.index, 'config'] == 'full'].iloc[0])
abl['p_holm'] = np.nan
for _, idx in abl[abl['config'] != 'full'].groupby(['exp', 'label', 'frame']).groups.items():
    abl.loc[idx, 'p_holm'] = au.holm(abl.loc[idx, 'p_boot'])
abl.to_csv(RESULTS / 'ablation_feature_families.csv', index=False)
abl[abl['frame'] == 'transfer'].round(3)
"""),
    code("""
# H6 verdict per experiment: does any 'drop' configuration match or beat the full set?
drop = abl[abl['config'].str.startswith('drop')]
verdict = (drop.groupby(['exp', 'label', 'frame'])
           .apply(lambda g: pd.Series({
               'best_drop': g.loc[g['delta_vs_full'].idxmax(), 'config'],
               'best_drop_delta': g['delta_vs_full'].max(),
               'drops_at_or_above_full': int((g['delta_vs_full'] >= 0).sum()),
               'drops_significantly_below_full': int(((g['delta_vs_full'] < 0) & (g['p_holm'] < 0.05)).sum()),
               'H6_supported': bool((g['delta_vs_full'] < 0).all())}))
           .reset_index())
verdict
"""),
    code("""
# Heatmap: change in macro F1 when a family is removed
for frame in ['indomain', 'transfer']:
    h = (drop[(drop['frame'] == frame) & (drop['label'] == LABELS[0])]
         .pivot(index='config', columns='exp', values='delta_vs_full'))
    h.index = h.index.str.replace('drop: ', '')
    fig, ax = plt.subplots(figsize=(8, 4.5))
    lim = np.nanmax(np.abs(h.values)) or 0.01
    im = ax.imshow(h.values, cmap='RdBu', vmin=-lim, vmax=lim, aspect='auto')
    ax.set_xticks(range(h.shape[1])); ax.set_xticklabels(h.columns, rotation=20, ha='right')
    ax.set_yticks(range(h.shape[0])); ax.set_yticklabels(h.index)
    for i in range(h.shape[0]):
        for j in range(h.shape[1]):
            ax.text(j, i, f'{h.values[i, j]:+.2f}', ha='center', va='center', fontsize=8)
    fig.colorbar(im, label='macro F1 minus full set')
    ax.set_title(f'Feature family removed ({frame}, {LABELS[0]})')
    plt.tight_layout(); plt.savefig(FIGURES / f'fig_ablation_{frame}.pdf'); plt.show()
"""),
])

# ======================================================================================
# 03 error analysis by device type
# ======================================================================================
write("03_error_analysis_by_device.ipynb", [
    md("""
# 03 Error analysis by device type and condition

Which device types and conditions produce the misclassifications? Based on the exported
predictions of the three class random forest (no training).

1. Per device type and group: where do its packets go (background, own group, other group)?
2. The same per address (majority vote): how many addresses of each device are found?
3. Background addresses that are predicted as a group (false positives) by manufacturer.
4. Errors over time (two hour blocks in domain, 10 minute windows on the university capture).
"""),
    code(SETUP),
    code("""
MODEL = 'random_forest'
SPLIT = 'grouped'
LABEL = 'multi'
"""),
    md("## 1. Packet outcome per device type and group"),
    code("""
rows = []
for e in EXPS:
    for frame in ['indomain', 'transfer']:
        p = au.load_predictions(e, LABEL, MODEL, SPLIT, frame)
        tgt = p[p['y_true'] != 0]
        for (dev, grp), g in tgt.groupby(['labelled_device', 'y_true']):
            rows.append(dict(exp=au.exp_name(e), frame=frame, device=dev, group=int(grp),
                             packets=len(g), sources=g['source'].nunique(),
                             own_group=(g['y_pred'] == grp).mean(),
                             other_group=((g['y_pred'] != grp) & (g['y_pred'] != 0)).mean(),
                             background=(g['y_pred'] == 0).mean()))
dev = pd.DataFrame(rows)
dev.to_csv(RESULTS / 'error_by_device.csv', index=False)
dev[dev['frame'] == 'transfer'].round(3)
"""),
    code("""
# Recall of the own group per device type on the university capture
h = (dev[dev['frame'] == 'transfer'].assign(unit=lambda d: d['device'] + ' G' + d['group'].astype(str))
     .pivot(index='unit', columns='exp', values='own_group'))
fig, ax = plt.subplots(figsize=(9, 0.45 * len(h) + 1.5))
im = ax.imshow(h.values, cmap='viridis', vmin=0, vmax=1, aspect='auto')
ax.set_xticks(range(h.shape[1])); ax.set_xticklabels(h.columns, rotation=20, ha='right')
ax.set_yticks(range(h.shape[0])); ax.set_yticklabels(h.index)
for i in range(h.shape[0]):
    for j in range(h.shape[1]):
        if not np.isnan(h.values[i, j]):
            ax.text(j, i, f'{h.values[i, j]:.2f}', ha='center', va='center', fontsize=8,
                    color='white' if h.values[i, j] < 0.5 else 'black')
fig.colorbar(im, label='share of packets assigned to the own group')
ax.set_title('University capture, random forest, grouped split')
plt.tight_layout(); plt.savefig(FIGURES / 'fig_error_by_device_transfer.pdf'); plt.show()
"""),
    md("## 1b. The same per sublabel (e.g. phone scan requests against adverts)"),
    code("""
rows = []
for e in EXPS:
    p = au.load_predictions(e, LABEL, MODEL, SPLIT, 'transfer')
    tgt = p[p['y_true'] != 0].copy()
    tgt['sublabel_device'] = tgt['sublabel_device'].fillna('(none)')
    for (dev_, sub, grp), g in tgt.groupby(['labelled_device', 'sublabel_device', 'y_true']):
        rows.append(dict(exp=au.exp_name(e), device=dev_, sublabel=sub, group=int(grp), packets=len(g),
                         own_group=(g['y_pred'] == grp).mean(),
                         other_group=((g['y_pred'] != grp) & (g['y_pred'] != 0)).mean(),
                         background=(g['y_pred'] == 0).mean()))
pd.DataFrame(rows).round(3)
"""),
    md("## 2. Per address (majority vote)"),
    code("""
rows = []
for e in EXPS:
    for frame in ['indomain', 'transfer']:
        p = au.load_predictions(e, LABEL, MODEL, SPLIT, frame)
        src = au.majority_vote(p, p['y_true'], p['y_pred'])
        for (d_, grp), g in src[src['y_true'] != 0].groupby(['labelled_device', 'y_true']):
            rows.append(dict(exp=au.exp_name(e), frame=frame, device=d_, group=int(grp), addresses=len(g),
                             correct=(g['y_pred'] == grp).sum(),
                             other_group=((g['y_pred'] != grp) & (g['y_pred'] != 0)).sum(),
                             background=(g['y_pred'] == 0).sum(),
                             median_packets_per_address=g['n_packets'].median()))
src_dev = pd.DataFrame(rows)
src_dev.to_csv(RESULTS / 'error_by_device_sources.csv', index=False)
src_dev[src_dev['frame'] == 'transfer']
"""),
    md("## 3. False positives: background addresses predicted as a group"),
    code("""
rows = []
for e in EXPS:
    p = au.load_predictions(e, LABEL, MODEL, SPLIT, 'transfer')
    ev = au.load_frame(e, 'eval', columns=['company_id'])
    p['company_id'] = ev['company_id'].to_numpy()[p['row']]
    bg = p[p['y_true'] == 0]
    fp = bg[bg['y_pred'] != 0]
    by = (fp.groupby('company_id', dropna=False)
          .agg(fp_packets=('y_pred', 'size'), fp_sources=('source', 'nunique'),
               as_g1=('y_pred', lambda s: (s == 1).sum()), as_g2=('y_pred', lambda s: (s == 2).sum()))
          .sort_values('fp_packets', ascending=False).head(10).reset_index())
    by.insert(0, 'exp', au.exp_name(e))
    by['share_of_all_fp'] = by['fp_packets'] / max(len(fp), 1)
    by['bg_fp_rate'] = len(fp) / max(len(bg), 1)
    rows.append(by)
fps = pd.concat(rows, ignore_index=True)
fps.to_csv(RESULTS / 'false_positives_by_company.csv', index=False)
fps
"""),
    md("## 4. Errors over time"),
    code("""
rows = []
for e in EXPS:
    p = au.load_predictions(e, LABEL, MODEL, SPLIT, 'indomain')
    for b, g in p.groupby('block'):
        t = g[g['y_true'] != 0]
        rows.append(dict(exp=au.exp_name(e), frame='indomain', window=f'block {int(b)}',
                         target_packets=len(t), own_group=(t['y_pred'] == t['y_true']).mean() if len(t) else np.nan))
    p = au.load_predictions(e, LABEL, MODEL, SPLIT, 'transfer')
    p['window'] = ((p['time'] - p['time'].min()) // 600).astype(int)
    for w, g in p.groupby('window'):
        t = g[g['y_true'] != 0]
        rows.append(dict(exp=au.exp_name(e), frame='transfer', window=f'{10 * w:03d}-{10 * w + 10:03d} min',
                         target_packets=len(t), own_group=(t['y_pred'] == t['y_true']).mean() if len(t) else np.nan))
over_time = pd.DataFrame(rows)
over_time.to_csv(RESULTS / 'error_over_time.csv', index=False)
over_time.pivot_table(index=['frame', 'window'], columns='exp', values='own_group').round(2)
"""),
])

# ======================================================================================
# 04 feature importance per family
# ======================================================================================
write("04_feature_importance_per_family.ipynb", [
    md("""
# 04 Feature importance per feature family

Reads the importance tables written by the modelling notebooks (one per run). For the random
forest these are the impurity based importances, for MLP and SGD the permutation importances on
the in domain test split. Importances are summed per family and normalised to a share per run.

Note: impurity importance favours features with many distinct values (here the per source
timing statistics) and is computed on the training capture, so it says what the model *uses*,
not what *transfers*. Notebook 02 (ablation) answers the second question.
"""),
    code(SETUP),
    code("""
rows = []
for e in EXPS:
    for f in sorted((au.ARTIFACT_DIR / e / 'importance').glob('*.csv')):
        label, model, split, frame = f.stem.split('__')
        d = pd.read_csv(f)
        d = d.assign(exp=au.exp_name(e), label=label, model=model, split=split)
        rows.append(d)
imp = pd.concat(rows, ignore_index=True)
imp['family'] = imp['Feature'].map(au.family_of)
imp.to_csv(RESULTS / 'feature_importance_all.csv', index=False)

fam = (imp.groupby(['exp', 'label', 'model', 'split', 'family'])['Importance'].sum().reset_index())
fam['share'] = fam['Importance'] / fam.groupby(['exp', 'label', 'model', 'split'])['Importance'].transform(
    lambda s: s.clip(lower=0).sum())
fam['n_features'] = fam['family'].map(lambda f: len(au.FEATURE_FAMILIES.get(f, [])))
fam['share_per_feature'] = fam['share'] / fam['n_features'].replace(0, np.nan)
fam.to_csv(RESULTS / 'feature_importance_per_family.csv', index=False)
fam[(fam['model'] == 'random_forest') & (fam['label'] == 'multi') & (fam['split'] == 'grouped')] \\
    .pivot(index='family', columns='exp', values='share').round(3)
"""),
    code("""
for label in ['multi', 'binary']:
    h = (fam[(fam['model'] == 'random_forest') & (fam['label'] == label) & (fam['split'] == 'grouped')]
         .pivot(index='family', columns='exp', values='share'))
    if h.empty:
        continue
    h = h.loc[h.mean(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    im = ax.imshow(h.values, cmap='Blues', aspect='auto', vmin=0)
    ax.set_xticks(range(h.shape[1])); ax.set_xticklabels(h.columns, rotation=20, ha='right')
    ax.set_yticks(range(h.shape[0])); ax.set_yticklabels(h.index)
    for i in range(h.shape[0]):
        for j in range(h.shape[1]):
            ax.text(j, i, f'{h.values[i, j]:.2f}', ha='center', va='center', fontsize=8)
    fig.colorbar(im, label='share of the total importance')
    ax.set_title(f'Random forest importance per family ({label}, grouped split)')
    plt.tight_layout(); plt.savefig(FIGURES / f'fig_importance_family_{label}.pdf'); plt.show()
"""),
    md("## Top ten features per experiment (random forest, three class, grouped split)"),
    code("""
top = (imp[(imp['model'] == 'random_forest') & (imp['label'] == 'multi') & (imp['split'] == 'grouped')]
       .sort_values('Importance', ascending=False).groupby('exp').head(10))
top[['exp', 'Feature', 'family', 'Importance']].reset_index(drop=True)
"""),
    md("## Agreement between learners (MLP and SGD use permutation importance)"),
    code("""
fam[(fam['label'] == 'multi') & (fam['split'] == 'grouped')] \\
    .pivot_table(index='family', columns=['model'], values='share', aggfunc='mean').round(3)
"""),
])

# ======================================================================================
# 05 clustering and payload side study
# ======================================================================================
write("05_clustering_and_payload.ipynb", [
    md("""
# 05 Clustering and payload side study (evidence for the decision)

Both are open decisions (report in results, appendix, or drop). This notebook produces the
numbers for that decision on all six experiments from the exported frames.

**Clustering.** Per address aggregates (mean, std, min, max of the model features, addresses
with at least `MIN_PACKETS` packets), standardised, k-means / BIRCH / bisecting k-means.
Two questions:
* all addresses, k = 3: do background, group 1 and group 2 form clusters?
* target addresses only, k = 2: does unsupervised clustering separate group 1 from group 2?
  This is the actual multi-group question and the one worth reporting.

Note: the clustering cells in the modelling notebooks keep `group_label_multi` and `group_id`
in the feature matrix (only `source` and `group_label` are dropped), so their ARI includes the
label itself. Here only model features are used.

**Payload.** For each target unit and byte position of the manufacturer data: is the byte
constant for that unit, does it differ between the two units of the same model (group 1 against
group 2), and is it the same in the training capture and the university capture? A byte that
satisfies all three identifies the unit across sessions. This is the upper bound described in
the design chapter, not a model input.
"""),
    code(SETUP),
    code("""
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, Birch, BisectingKMeans
from sklearn.metrics import adjusted_rand_score, v_measure_score

MIN_PACKETS = 36
AGG = ['mean', 'std', 'min', 'max']

def aggregate(df, feats):
    counts = df['source'].value_counts()
    keep = counts[counts >= MIN_PACKETS].index
    d = df[df['source'].isin(keep)]
    X = d.groupby('source')[feats].agg(AGG)
    X.columns = ['_'.join(c) for c in X.columns]
    y = d.groupby('source')['group_label_multi'].agg(lambda s: s.value_counts().idxmax())
    dev_ = d.groupby('source')['labelled_device'].first()
    return X.fillna(0), y, dev_

def cluster(X, k):
    Xs = StandardScaler().fit_transform(X)
    return {'kmeans': KMeans(n_clusters=k, n_init=10, random_state=1).fit_predict(Xs),
            'birch': Birch(n_clusters=k).fit_predict(Xs),
            'bisecting_kmeans': BisectingKMeans(n_clusters=k, random_state=1).fit_predict(Xs)}
"""),
    code("""
rows = []
for e in EXPS:
    for which in ['train', 'eval']:
        df = au.load_frame(e, which)
        feats = au.feature_columns(df)
        X, y, dev_ = aggregate(df, feats)
        for scope, mask, k in [('all addresses, k=3', np.ones(len(y), bool), 3),
                               ('targets only, k=2', (y != 0).to_numpy(), 2)]:
            if mask.sum() < k + 1:
                continue
            for algo, lab in cluster(X[mask], k).items():
                rows.append(dict(exp=au.exp_name(e), capture=which, scope=scope, algorithm=algo,
                                 addresses=int(mask.sum()),
                                 targets=int((y[mask] != 0).sum()),
                                 ari=adjusted_rand_score(y[mask], lab),
                                 v_measure=v_measure_score(y[mask], lab)))
clus = pd.DataFrame(rows)
clus.to_csv(RESULTS / 'clustering.csv', index=False)
clus.pivot_table(index=['scope', 'algorithm', 'capture'], columns='exp', values='ari').round(3)
"""),
    md("""
Reading guide: ARI 0 is chance, 1 is a perfect match. If the *targets only* ARI stays near 0
in the equal distance experiments, clustering cannot separate the two groups either, which is
a one line result for the thesis (appendix), not a section.
"""),
    md("## Payload: unit specific constant bytes"),
    code("""
def byte_table(df):
    # one row per target packet with the manufacturer data split into byte columns
    t = df[(df['group_label_multi'] != 0) & df['data'].notna()][['labelled_device', 'group_label_multi', 'data']].copy()
    t['data'] = t['data'].astype(str)
    t = t[t['data'].str.fullmatch(r'[0-9a-fA-F]+') & (t['data'].str.len() % 2 == 0)]
    t['len'] = t['data'].str.len() // 2
    return t

DOMINANCE = 0.95   # a byte counts as constant if one value covers at least 95 % of the unit's packets
                   # (a few packets per unit are other frame variants or label noise)

def constant_bytes(t):
    # (device, group, length) -> {byte position: value} for positions that (almost) never change
    out = {}
    for (dev_, grp, ln), g in t.groupby(['labelled_device', 'group_label_multi', 'len']):
        if len(g) < 50:
            continue
        b = np.array([list(bytes.fromhex(s)) for s in g['data'].to_numpy()])
        const = {}
        for i in range(b.shape[1]):
            vals, cnt = np.unique(b[:, i], return_counts=True)
            if cnt.max() / len(b) >= DOMINANCE:
                const[i] = int(vals[cnt.argmax()])
        out[(dev_, int(grp), int(ln))] = const
    return out

rows = []
for e in EXPS:
    ct = constant_bytes(byte_table(au.load_frame(e, 'train', columns=['labelled_device', 'group_label_multi', 'data'])))
    ce = constant_bytes(byte_table(au.load_frame(e, 'eval', columns=['labelled_device', 'group_label_multi', 'data'])))
    for (dev_, grp, ln), const in ct.items():
        other = ct.get((dev_, 3 - grp, ln), {})
        ev_ = ce.get((dev_, grp, ln), {})
        for pos, val in const.items():
            if pos in other and other[pos] != val:      # differs between the two units of this model
                rows.append(dict(exp=au.exp_name(e), device=dev_, group=grp, payload_len=ln, byte=pos,
                                 value=f'{val:02x}', other_group_value=f'{other[pos]:02x}',
                                 same_in_university_capture=ev_.get(pos) == val))
payload = pd.DataFrame(rows, columns=['exp', 'device', 'group', 'payload_len', 'byte', 'value',
                                      'other_group_value', 'same_in_university_capture'])
payload.to_csv(RESULTS / 'payload_unit_bytes.csv', index=False)
payload if len(payload) else 'no unit specific constant byte found'
"""),
    code("""
# Summary: devices with at least one byte that identifies the unit across both sessions
if len(payload):
    (payload.groupby(['exp', 'device'])
     .agg(distinguishing_bytes=('byte', 'nunique'),
          stable_across_sessions=('same_in_university_capture', 'sum'))
     .reset_index())
"""),
])
