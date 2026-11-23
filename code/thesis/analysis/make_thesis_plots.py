"""Build every figure used in the thesis into plots/Thesis.

Run from anywhere:  python code/thesis/analysis/make_thesis_plots.py

Inputs (nothing is trained or recomputed here):
  - code/thesis/analysis/results/*.csv                (written by the analysis notebooks 01 to 05)
  - code/thesis/analysis/results/figures/*.pdf         (written by the analysis notebooks, copied as they are)
  - the stored plot outputs of the executed modelling notebooks (confusion matrices, random forest
    feature importance, learning curves), combined into one grid per figure

Output: plots/Thesis/{design,results,analysis}/*.pdf
"""
import base64
import io
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[3]
MODELING = ROOT / 'code' / 'thesis' / 'modeling'
RESULTS = ROOT / 'code' / 'thesis' / 'analysis' / 'results'
OUT = ROOT / 'plots' / 'Thesis'

EXP = [('E1', 'apple same', 'modeling_10h2h_IP12_AAP1P_ST_SameDist.ipynb'),
       ('E2', 'apple diff', 'modeling_10h2h_IP12_AAP1P_ST_DiffDist.ipynb'),
       ('E3', 'samsung same', 'modeling_10h2h_sa54_sbfe_aat_sameDist.ipynb'),
       ('E4', 'samsung diff', 'modeling_10h2h_sa54_sbfe_aat_diffDist.ipynb'),
       ('E5', 'mixed same', 'modeling_6h2h_A54FEAirtags_IP12PodsST_SameDist.ipynb'),
       ('E6', 'mixed diff', 'modeling_6h2h_A54FEAirtags_IP12PodsST_DiffDist.ipynb')]
EXP_NAMES = [f'{e} {d}' for e, d, _ in EXP]
MODELS = ['Random Forest', 'Multi-layer Perceptron', 'Stochastic Gradient Descent']
SHORT = {'Random Forest': 'RF', 'Multi-layer Perceptron': 'MLP', 'Stochastic Gradient Descent': 'SGD'}

# Categorical slots 1 to 4 (checked for colour vision deficiency and normal vision separation).
COL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']
GREY = '#9a9a9a'
INK, INK2, GRID = '#222222', '#555555', '#dddddd'
CHANCE = 1 / 3

plt.rcParams.update({'font.family': 'serif', 'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False,
                     'axes.spines.right': False, 'legend.frameon': False, 'pdf.fonttype': 42})

TICKS = [e + '\n' + d.replace(' ', '\n') for e, d, _ in EXP]


def save(fig, sub, name, dpi=None):
    path = OUT / sub / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print('wrote', path.relative_to(ROOT))


def style(ax, ylabel=None, ylim=(0, 1.05)):
    ax.set_ylim(*ylim)
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    if ylabel:
        ax.set_ylabel(ylabel)


def bars(ax, values, names, colors):
    n, x = len(names), np.arange(len(EXP))
    w = 0.8 / n
    for i, (name, c) in enumerate(zip(names, colors)):
        xs = x + (i - (n - 1) / 2) * w
        ax.bar(xs, values[i], w * 0.92, color=c, label=name, zorder=2)
        for xi, vi in zip(xs, values[i]):
            ax.text(xi, vi + 0.015, f'{vi:.2f}', ha='center', va='bottom', fontsize=6.5, color=INK2)
    ax.axhline(CHANCE, ls='--', lw=1, color=INK2, zorder=1)
    ax.set_xticks(x, TICKS)
    style(ax)


# ------------------------------------------------------------------------------------------------
# Design
# ------------------------------------------------------------------------------------------------
def experiment_matrix():
    rows = [('Apple set', 'identical hardware', '2x iPhone 12, 2x AirPods Pro, 2x SmartTag', '10 h (5 x 2 h)', ['E1', 'E2']),
            ('Samsung set', 'identical hardware', '2x Galaxy A54, 2x Buds FE, 2x AirTag', '10 h (5 x 2 h)', ['E3', 'E4']),
            ('Mixed set', 'different hardware', 'G1: Galaxy A54, Buds FE, AirTag\nG2: iPhone 12, AirPods Pro, SmartTag',
             '6 h (3 x 2 h)', ['E5', 'E6'])]
    cols = ['Equal distance', 'Different distance']
    fig, ax = plt.subplots(figsize=(10, 5.4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6.2); ax.axis('off')
    x0, cw, rh = 2.6, 3.7, 1.75
    for j, c in enumerate(cols):
        ax.text(x0 + j * cw + cw / 2, 5.85, c, ha='center', va='center', fontsize=10, weight='bold', color=INK)
    for i, (name, hw, devices, train, ids) in enumerate(rows):
        y = 5.4 - (i + 1) * rh
        mixed = hw.startswith('different')
        if mixed:  # separator between the identical and the different hardware configuration
            ax.plot([0.1, x0 + 2 * cw], [y + rh + 0.03] * 2, color=INK2, lw=1, ls=(0, (4, 3)))
        ax.text(0.15, y + rh * 0.68, name, fontsize=10, weight='bold', color=INK, va='center')
        ax.text(0.15, y + rh * 0.42, hw, fontsize=8, color=COL[1] if mixed else INK2, va='center', style='italic')
        ax.text(0.15, y + rh * 0.18, f'train {train}', fontsize=8, color=INK2, va='center')
        for j, eid in enumerate(ids):
            face = '#fdf0ea' if mixed else '#eef4fc'
            edge = COL[1] if mixed else COL[0]
            ax.add_patch(FancyBboxPatch((x0 + j * cw + 0.08, y + 0.08), cw - 0.16, rh - 0.16,
                                        boxstyle='round,pad=0,rounding_size=0.08', fc=face, ec=edge, lw=1.2))
            cx = x0 + j * cw + cw / 2
            ax.text(cx, y + rh * 0.74, eid, ha='center', va='center', fontsize=12, weight='bold', color=INK)
            ax.text(cx, y + rh * 0.44, devices, ha='center', va='center', fontsize=7.5, color=INK, linespacing=1.3)
            ax.text(cx, y + rh * 0.16, 'eval: 2 h university capture', ha='center', va='center', fontsize=7.5, color=INK2)
    save(fig, 'design', 'fig_experiment_matrix.pdf')


FAMILIES = [  # (family, [(feature, added in this thesis)]) in the order of the design chapter table
    ('PDU role', [(f, False) for f in ['is_ADV_IND', 'is_ADV_NONCONN_IND', 'is_ADV_SCAN_IND', 'is_ADV_EXT_IND',
                                       'is_AUX_ADV_IND', 'is_SCAN_REQ']]),
    ('Channel and PHY', [(f, False) for f in ['is_37', 'is_38', 'is_39', 'is_adv_channel', 'is_broadcast']]),
    ('Payload shape', [('has_company_id', True), ('has_uuid16', True), ('service_data_len_ratio', False)]),
    ('Burst microstructure', [(f, False) for f in ['packets_within_2ms', 'packets_within_25ms', 'packets_within_100ms',
                                                   'packets_within_1s']]),
    ('Joint activity', [(f, True) for f in ['bin_total_packets', 'bin_distinct_sources', 'coactive_packets_in_bin',
                                            'coactive_sources_in_bin', 'coactive_density', 'is_solo_bin', 'bin_sync_flips']]),
    ('Payload dynamics', [(f, True) for f in ['payload_changed', 'payload_sync_in_bin', 'payload_sync_density']]),
    ('Timing', [(f, True) for f in ['iat_median', 'iat_mean', 'iat_std', 'iat_deviation', 'is_burst_iat']]),
    ('Shape and session span', [(f, False) for f in ['len_ad_type_binned', 'len_data_binned', 'time_start_end_binned',
                                                     'delta_end_start_binned', 'delta_start_start_binned',
                                                     'packet_send_freq_binned', 'time_to_next_binned',
                                                     'data_len_ratio_binned', 'ad_type_len_ratio_binned']]),
]
NEW = COL[1]


def feature_map():
    fam = dict(FAMILIES)
    assert sum(len(v) for v in fam.values()) == 42 and sum(n for v in fam.values() for _, n in v) == 17
    fig, ax = plt.subplots(figsize=(12, 8.4))
    ax.set_xlim(0, 120); ax.set_ylim(0, 84); ax.axis('off')

    def box(x, y, w, h, name, anchor=None, target=None):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.8', fc='#f6f8fb', ec='#9aa5b4', lw=0.9))
        feats = fam[name]
        ax.text(x + 1.2, y + h - 1.6, f'{name}  ({len(feats)})', fontsize=8.5, weight='bold', color=INK, va='top')
        for k, (f, new) in enumerate(feats):
            ax.text(x + 1.2, y + h - 4.6 - k * 1.75, f, fontsize=6.8, family='monospace', va='top',
                    color=NEW if new else INK, weight='bold' if new else 'normal')
        if anchor and target:
            ax.plot([anchor[0], target[0]], [anchor[1], target[1]], ls=(0, (3, 2)), lw=0.8, color='#8a94a3', zorder=0)

    # --- capture timeline -----------------------------------------------------------------------
    ax.text(2, 83, 'Capture timeline', fontsize=10, weight='bold', color=INK, va='top')
    ax.text(2, 80.6, 'packets of one source (blue, shade = payload) and of other sources (grey)', fontsize=7.5,
            color=INK2, style='italic', va='top')
    lane_a, lane_b, pw, ph = 73.0, 65.4, 2.2, 2.6
    xs_a = [8, 17, 26, 35, 52, 61, 70, 87, 96, 105]
    shade = ['#cfe0f6'] * 4 + ['#7fb0ea'] * 6  # payload changes after the fourth packet
    for x, c in zip(xs_a, shade):
        ax.add_patch(plt.Rectangle((x, lane_a), pw, ph, fc=c, ec=COL[0], lw=0.8))
    for x in [12, 21.5, 30, 44, 47, 57, 66, 78, 82, 92, 100, 110]:
        ax.add_patch(plt.Rectangle((x, lane_b), pw * 0.8, ph * 0.8, fc='#d9d9d9', ec='#9a9a9a', lw=0.6))
    ax.text(4.5, lane_a + ph / 2, 'A', fontsize=8, color=COL[0], va='center', ha='center', weight='bold')
    ax.text(4.5, lane_b + ph * 0.4, 'other', fontsize=6.5, color=INK2, va='center', ha='center')
    ax.plot([2, 116], [lane_b - 1.2] * 2, color=INK2, lw=0.8)
    ax.annotate('', xy=(117, lane_b - 1.2), xytext=(115, lane_b - 1.2), arrowprops=dict(arrowstyle='->', color=INK2, lw=0.8))
    ax.text(117, lane_b - 2.6, 'time', fontsize=7, color=INK2, ha='right', va='top')
    # inter arrival time (Timing)
    ax.annotate('', xy=(17, lane_a + ph + 1.2), xytext=(8 + pw, lane_a + ph + 1.2),
                arrowprops=dict(arrowstyle='<->', color=INK, lw=0.8))
    ax.text(13.6, lane_a + ph + 1.8, 'interval to the previous packet', fontsize=6.8, color=INK, ha='center', va='bottom')
    # burst window (Burst microstructure)
    ax.annotate('', xy=(26 - 7, lane_a - 1.2), xytext=(26 + pw + 7, lane_a - 1.2),
                arrowprops=dict(arrowstyle='|-|', color=INK, lw=0.8, mutation_scale=2))
    ax.text(27, lane_a - 2.0, 'window around a packet', fontsize=6.8, color=INK, ha='center', va='top')
    # 500 ms bin (Joint activity)
    ax.add_patch(plt.Rectangle((42.5, lane_b - 0.8), 14, lane_a + ph - lane_b + 1.6, fc=COL[2], alpha=0.13, ec=COL[2], lw=0.8))
    ax.text(49.5, lane_a + ph + 1.8, '500 ms bin', fontsize=6.8, color=INK, ha='center', va='bottom')
    # payload change (Payload dynamics)
    ax.annotate('payload changes', xy=(52 + pw / 2, lane_a + ph + 0.2), xytext=(66, lane_a + ph + 2.6), fontsize=6.8,
                color=INK, ha='left', va='bottom', arrowprops=dict(arrowstyle='->', color=INK, lw=0.7))

    # --- time based families ------------------------------------------------------------------
    w, h, y = 27.5, 17, 44.5
    xs = [2, 31.5, 61, 90.5]
    box(xs[0], y, w, h, 'Timing', (xs[0] + w / 2, y + h), (13.6, lane_a + ph + 1.2))
    box(xs[1], y, w, h, 'Burst microstructure', (xs[1] + w / 2, y + h), (27, lane_a - 4.0))
    box(xs[2], y, w, h, 'Joint activity', (xs[2] + w / 2, y + h), (49.5, lane_b - 0.8))
    box(xs[3], y, w, h, 'Payload dynamics', (xs[3] + w / 2, y + h), (70, lane_a + ph + 2.5))

    # --- one advertising PDU --------------------------------------------------------------------
    py, phh = 29, 6.5
    ax.text(2, py + phh + 2.6, 'One advertising PDU (legacy, 1 Mbit/s)', fontsize=10, weight='bold', color=INK, va='bottom')
    fields = [('Preamble', '1 B', 9, '#eef1f6'), ('Access\nAddress', '4 B', 14, '#e6ebf3'), ('PDU\nheader', '2 B', 12, '#dfe6f0'),
              ('AdvA\n(sender address)', '6 B', 18, '#f6eaea'), ('AdvData', '0 to 31 B', 38, '#eaf3ea'), ('CRC', '3 B', 10, '#eef1f6')]
    x, pos = 4, {}
    for name, size, fw, fc in fields:
        ax.add_patch(plt.Rectangle((x, py), fw, phh, fc=fc, ec=INK, lw=0.8))
        ax.text(x + fw / 2, py + phh * 0.62, name, fontsize=7.5, ha='center', va='center', color=INK, linespacing=1.1)
        ax.text(x + fw / 2, py + 1.0, size, fontsize=6.5, ha='center', va='center', color=INK2)
        pos[name.split('\n')[0]] = (x, fw)
        x += fw
    ax.add_patch(plt.Rectangle((4, py + phh), x - 4, 1.4, fc='#dbe7f7', ec=COL[0], lw=0.8))
    ax.text((4 + x) / 2, py + phh + 0.7, 'sent on advertising channel 37, 38 or 39', fontsize=6.8, ha='center', va='center', color=COL[0])
    ax.text(x + 1, py + phh + 0.7, '<- Channel and PHY', fontsize=6.8, ha='left', va='center', color=COL[0], weight='bold')
    # AD structures inside AdvData
    ax0, aw = pos['AdvData']
    sub = [('Length', '1 B', 0.22), ('AD Type', '1 to 2 B', 0.33), ('AD Data', '0 to 29 B', 0.45)]
    sx = ax0
    for name, size, frac in sub:
        ax.add_patch(plt.Rectangle((sx, py - 6.2), aw * frac, 4.2, fc='#f4f9f4', ec=INK, lw=0.6, ls='--'))
        ax.text(sx + aw * frac / 2, py - 3.4, name, fontsize=6.8, ha='center', va='center', color=INK)
        ax.text(sx + aw * frac / 2, py - 5.2, size, fontsize=6, ha='center', va='center', color=INK2)
        sx += aw * frac
    ax.plot([ax0, ax0], [py, py - 2.0], color=INK2, lw=0.6, ls=':')
    ax.plot([ax0 + aw, ax0 + aw], [py, py - 2.0], color=INK2, lw=0.6, ls=':')
    ax.text(ax0 + aw + 1, py - 4.1, 'AD structures\ninside AdvData', fontsize=6.5, color=INK2, va='center')
    # not used
    a0, a_w = pos['AdvA']
    ax.text(a0 + a_w / 2, py - 2.4, 'not a feature:\naddress rotates;\nRSSI also excluded', fontsize=6.5, color='#b5452b',
            ha='center', va='top', style='italic', linespacing=1.15)

    # --- packet based families ----------------------------------------------------------------
    y2 = 1.0
    h0, w0 = pos['PDU']
    box(xs[0], y2, w, 15, 'PDU role', (xs[0] + w / 2, y2 + 15), (h0 + w0 / 2, py))
    box(xs[1], y2, w, 15, 'Channel and PHY')
    box(xs[2], y2, w, 15, 'Payload shape', (xs[2] + w / 2, y2 + 15), (pos['AdvData'][0] + aw * 0.4, py - 6.2))
    box(xs[3], y2, w, 20.5, 'Shape and session span', (xs[3] + w / 2, y2 + 20.5), (pos['AdvData'][0] + aw * 0.85, py - 6.2))

    # legend
    ax.text(119, 83, '42 features: 25 from the foundational work,', fontsize=7.5, color=INK, ha='right', va='top')
    ax.text(119, 80.6, '17 added in this thesis (bold, orange)', fontsize=7.5, color=NEW, weight='bold', ha='right', va='top')
    save(fig, 'design', 'fig_feature_map.pdf')


# ------------------------------------------------------------------------------------------------
# Results
# ------------------------------------------------------------------------------------------------
def results_bars():
    # metrics_all_runs.csv holds the packet level scores of every run (analysis notebook 01).
    res = pd.read_csv(RESULTS / 'metrics_all_runs.csv')
    res = res[res.exp.isin(EXP_NAMES) & (res.label == 'multi')]

    def get(frame, split, col, model):
        return [res[(res.exp == n) & (res.model == model) & (res.split == split) & (res.frame == frame)][col].iloc[0]
                for n in EXP_NAMES]

    fig, axes = plt.subplots(2, 2, figsize=(11, 6.4), sharey=True)
    for r, (frame, title) in enumerate([('indomain', 'In domain test split'), ('transfer', 'Transfer to the university capture')]):
        for c, split in enumerate(['packet', 'grouped']):
            ax = axes[r, c]
            bars(ax, [get(frame, split, 'macro_f1', m) for m in MODELS], [SHORT[m] for m in MODELS], COL)
            ax.set_title(f'{title}, {split} split', fontsize=9, color=INK, loc='left')
            if c == 0:
                ax.set_ylabel('Macro F1')
    axes[0, 0].legend(ncol=3, loc='lower left', bbox_to_anchor=(0, 1.12))
    fig.tight_layout()
    save(fig, 'results', 'fig_results_macro_f1.pdf')

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), sharey=True)
    for ax, split in zip(axes, ['packet', 'grouped']):
        bars(ax, [get('transfer', split, k, 'Random Forest') for k in ['f1_0', 'f1_1', 'f1_2']],
             ['Background', 'Group 1', 'Group 2'], COL)
        ax.set_title(f'Random forest, transfer, {split} split', fontsize=9, color=INK, loc='left')
    axes[0].set_ylabel('F1 score')
    axes[0].legend(ncol=3, loc='lower left', bbox_to_anchor=(0, 1.1))
    fig.tight_layout()
    save(fig, 'results', 'fig_results_class_f1_transfer.pdf')


# ------------------------------------------------------------------------------------------------
# Grids of the plots stored in the executed modelling notebooks
# ------------------------------------------------------------------------------------------------
def notebook_plots(nbfile):
    """Return {(label, model, grouped, kind): png} for CODE FIELD 6 and the learning curve of CODE FIELD 8."""
    nb = json.load(open(MODELING / nbfile, encoding='utf-8'))
    out = {}
    for cell in nb['cells']:
        src = ''.join(cell['source'])
        if '#        CODE FIELD 6        #' in src:
            run, phase = None, None
            for o in cell.get('outputs', []):
                if o['output_type'] == 'stream':
                    for line in ''.join(o['text']).splitlines():
                        if 'STARTING MODEL:' in line:
                            model = line.split('STARTING MODEL:')[1].split(' WITH')[0].strip()
                            grouped = 'SPLIT MODE: True' in line
                            label = line.rsplit('LABEL:', 1)[1].strip()
                            run, phase = (label, model, grouped), 'train'
                        elif 'Evaluation on Test Dataset' in line:
                            phase = 'eval'
                elif 'image/png' in o.get('data', {}):
                    is_cm = '2 Axes' in o['data']['text/plain'][0]
                    kind = f'cm_{phase}' if is_cm else 'importance'
                    out.setdefault(run + (kind,), o['data']['image/png'])
        if '#        CODE FIELD 8        #' in src:
            for o in cell.get('outputs', []):
                if 'image/png' in o.get('data', {}):
                    out['learning_curve'] = o['data']['image/png']
    return out


def grid(picks, name, ncol=3, size=(11, 6.6)):
    nrow = int(np.ceil(len(picks) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=size)
    for ax, (title, png) in zip(axes.flat, picks):
        ax.imshow(mpimg.imread(io.BytesIO(base64.b64decode(png)), format='png'))
        ax.axis('off')
        ax.set_title(title, fontsize=9, color=INK)
    for ax in list(axes.flat)[len(picks):]:
        ax.axis('off')
    fig.tight_layout()
    save(fig, 'results', name, dpi=200)


def results_grids():
    key = ('group_label_multi', 'Random Forest', True)
    plots = {f'{e} {d}': notebook_plots(f) for e, d, f in EXP}
    grid([(n, p[key + ('cm_eval',)]) for n, p in plots.items()], 'fig_results_cm_transfer.pdf')
    grid([(n, p[key + ('cm_train',)]) for n, p in plots.items()], 'fig_results_cm_indomain.pdf')
    grid([(n, p[key + ('importance',)]) for n, p in plots.items()], 'fig_results_rf_importance.pdf', ncol=2, size=(11, 10))
    grid([(n, p['learning_curve']) for n, p in plots.items()], 'fig_results_learning_curves.pdf', ncol=2, size=(11, 10))


# ------------------------------------------------------------------------------------------------
# New figures from the analysis tables
# ------------------------------------------------------------------------------------------------
SETS = [('Apple', 'E1 apple same', 'E2 apple diff'), ('Samsung', 'E3 samsung same', 'E4 samsung diff'),
        ('Mixed', 'E5 mixed same', 'E6 mixed diff')]


def error_over_time():
    t = pd.read_csv(RESULTS / 'error_over_time.csv')
    t = t[t.frame == 'transfer'].copy()
    t['start'] = t.window.str.slice(0, 3).astype(int)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True)
    for ax, (name, same, diff) in zip(axes, SETS):
        for exp, c, ls, lab in [(same, COL[0], '-', 'equal distance'), (diff, COL[1], '--', 'different distance')]:
            d = t[(t.exp == exp) & (t.target_packets >= 500)].sort_values('start')  # last partial window dropped
            ax.plot(d.start + 5, d.own_group, color=c, ls=ls, lw=2, marker='o', ms=4, label=f'{exp[:2]} {lab}')
        ax.set_title(f'{name} set', fontsize=9, color=INK, loc='left')
        ax.set_xlabel('Minutes into the university capture')
        ax.set_xticks(range(0, 140, 20))
        style(ax, 'Target packets in correct group' if ax is axes[0] else None)
        ax.legend(loc='lower left', fontsize=7.5)
    fig.tight_layout()
    save(fig, 'analysis', 'fig_error_over_time_transfer.pdf')


def false_positives_by_company():
    fp = pd.read_csv(RESULTS / 'false_positives_by_company.csv')
    keep = ['Apple, Inc.', 'Undefined', 'Samsung Electronics Co. Ltd.']
    fp['vendor'] = fp.company_id.where(fp.company_id.isin(keep), 'Other')
    share = fp.pivot_table(index='exp', columns='vendor', values='share_of_all_fp', aggfunc='sum').reindex(EXP_NAMES).fillna(0)
    share['Other'] = 1 - share[keep].sum(axis=1)  # the table lists the ten largest vendors only
    rate = fp.groupby('exp').bg_fp_rate.first().reindex(EXP_NAMES)
    g2 = (fp.groupby('exp').as_g2.sum() / fp.groupby('exp').fp_packets.sum()).reindex(EXP_NAMES)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={'width_ratios': [3, 2]})
    ax, x, bottom = axes[0], np.arange(len(EXP)), np.zeros(len(EXP))
    labels = {'Apple, Inc.': 'Apple', 'Undefined': 'Undefined', 'Samsung Electronics Co. Ltd.': 'Samsung', 'Other': 'Other'}
    # vendor colours differ from the group colours (slots 1 and 2) used in the right panel
    for v, c in zip(keep + ['Other'], [COL[2], COL[3], '#e87ba4', '#5c5c5c']):
        ax.bar(x, share[v], 0.62, bottom=bottom, color=c, label=labels[v], edgecolor='white', lw=1, zorder=2)
        bottom += share[v].values
    for xi, r in zip(x, rate):
        ax.text(xi, 1.02, f'{100 * r:.0f}%', ha='center', va='bottom', fontsize=7.5, color=INK2)
    ax.set_xticks(x, TICKS)
    style(ax, 'Share of false positives', (0, 1.12))
    ax.set_title('Vendor of background packets predicted as a group\n(number above bar: share of background predicted as a group)',
                 fontsize=9, color=INK, loc='left')
    ax.legend(ncol=4, loc='lower left', bbox_to_anchor=(0, -0.42), fontsize=8)

    ax = axes[1]
    ax.bar(x, 1 - g2, 0.62, color=COL[0], label='predicted as group 1', edgecolor='white', lw=1, zorder=2)
    ax.bar(x, g2, 0.62, bottom=1 - g2, color=COL[1], label='predicted as group 2', edgecolor='white', lw=1, zorder=2)
    for xi, v in zip(x, g2):
        ax.text(xi, 1 - v / 2, f'{100 * v:.0f}%', ha='center', va='center', fontsize=7.5, color='white')
    ax.set_xticks(x, [e for e, _, _ in EXP])
    style(ax, None, (0, 1.05))
    ax.set_title('Group the false positives are assigned to', fontsize=9, color=INK, loc='left')
    ax.legend(ncol=2, loc='lower left', bbox_to_anchor=(0, -0.3), fontsize=8)
    fig.tight_layout()
    save(fig, 'analysis', 'fig_false_positives_by_vendor.pdf')


def block_variability():
    b = pd.read_csv(RESULTS / 'block_variability.csv')
    b = b[b.label == 'group_label_multi']
    fig, ax = plt.subplots(figsize=(9, 3.4))
    rng = np.random.default_rng(1)  # fixed jitter so the figure is identical on every run
    for k, (frame, c, m, lab) in enumerate([('left out block', COL[0], 'o', 'left out block (in domain)'),
                                            ('transfer', COL[1], 's', 'university capture (transfer)')]):
        for i, n in enumerate(EXP_NAMES):
            v = b[(b.exp == n) & (b.frame == frame)].macro_f1.values
            xs = i + (k - 0.5) * 0.3 + rng.uniform(-0.05, 0.05, len(v))
            ax.scatter(xs, v, s=22, color=c, marker=m, zorder=3, label=lab if i == 0 else None, edgecolor='white', lw=0.6)
            ax.hlines(v.mean(), i + (k - 0.5) * 0.3 - 0.11, i + (k - 0.5) * 0.3 + 0.11, color=INK, lw=1.2, zorder=4)
    ax.axhline(CHANCE, ls='--', lw=1, color=INK2, zorder=1)
    ax.set_xticks(range(len(EXP)), TICKS)
    style(ax, 'Macro F1')
    ax.set_title('Random forest trained on four blocks, tested on the fifth block and on the university capture\n'
                 '(one marker per left out block, line: mean)', fontsize=9, color=INK, loc='left')
    ax.legend(loc='lower left', fontsize=8, ncol=2)
    fig.tight_layout()
    save(fig, 'analysis', 'fig_block_variability.pdf')


def distance_pairs():
    ci = pd.read_csv(RESULTS / 'confidence_intervals.csv')
    ci = ci[(ci.label == 'multi') & (ci.metric == 'macro_f1') & (ci.model == 'random_forest')]
    fig, axes = plt.subplots(2, 2, figsize=(10, 5.6), sharey='row')
    for r, frame in enumerate(['indomain', 'transfer']):
        for c, split in enumerate(['packet', 'grouped']):
            ax = axes[r, c]
            for i, (name, same, diff) in enumerate(SETS):
                for k, (exp, col, m, lab) in enumerate([(same, COL[0], 'o', 'equal distance'), (diff, COL[1], 's', 'different distance')]):
                    row = ci[(ci.exp == exp) & (ci.frame == frame) & (ci.split == split)].iloc[0]
                    x = i + (k - 0.5) * 0.3
                    ax.errorbar(x, row.point, yerr=[[row.point - row.ci_low], [row.ci_high - row.point]], fmt=m, color=col,
                                ms=6, capsize=3, lw=1.5, label=lab if i == 0 else None, zorder=3)
                    ax.text(x + 0.06, row.point, f'{row.point:.2f}', fontsize=7, color=INK2, va='center')
            ax.set_xticks(range(3), [s[0] for s in SETS])
            ax.set_xlim(-0.6, 2.6)
            ax.axhline(CHANCE, ls='--', lw=1, color=INK2, zorder=1)
            ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
            ax.set_title(f'{"In domain" if frame == "indomain" else "Transfer"}, {split} split', fontsize=9, color=INK, loc='left')
            if c == 0:
                ax.set_ylabel('Macro F1')
    axes[0, 0].set_ylim(0.8, 1.02)
    axes[1, 0].set_ylim(0.25, 0.7)
    axes[0, 0].legend(ncol=2, loc='lower left', bbox_to_anchor=(0, 1.12))
    fig.tight_layout()
    save(fig, 'analysis', 'fig_distance_pairs.pdf')


def copy_analysis_figures():
    for f in sorted((RESULTS / 'figures').glob('*.pdf')):
        dst = OUT / 'analysis' / f.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dst)
        print('copied', dst.relative_to(ROOT))


if __name__ == '__main__':
    experiment_matrix()
    feature_map()
    results_bars()
    results_grids()
    error_over_time()
    false_positives_by_company()
    block_variability()
    distance_pairs()
    copy_analysis_figures()
