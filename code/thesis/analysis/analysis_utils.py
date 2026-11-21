"""Shared helpers for the modelling notebooks and the analysis notebooks.

The modelling notebooks (code/thesis/modeling/modeling_*.ipynb) do the expensive part once:
loading, feature engineering, binning and balancing. They export the two balanced
frames and the per-packet predictions of every run to

    code/thesis/modeling/artifacts/<experiment_id>/
        train_balanced.parquet      balanced training frame (all columns, incl. source,
                                    labelled_device, sublabel_device, block)
        eval_balanced.parquet       balanced university capture frame
        predictions/<run>.parquet   one file per label x model x split x frame:
                                    row, y_true, y_pred, score_<class>...
        importance/<run>.csv        feature importance of that run
        metrics.csv                 one row per run with all metrics

Everything in code/thesis/analysis/*.ipynb only reads these files, so no analysis needs the
30 to 60 minute feature engineering again.
"""
from __future__ import annotations

import pathlib
import re

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    matthews_corrcoef,
)

# --------------------------------------------------------------------------------------
# Paths and experiment registry
# --------------------------------------------------------------------------------------
CODE_DIR = pathlib.Path(__file__).resolve().parents[1]
ARTIFACT_DIR = CODE_DIR / "modeling" / "artifacts"

# experiment id -> (short name used in the thesis, modelling notebook)
EXPERIMENTS = {
    "E1_apple_same": ("E1 apple same", "modeling_10h2h_IP12_AAP1P_ST_SameDist.ipynb"),
    "E2_apple_diff": ("E2 apple diff", "modeling_10h2h_IP12_AAP1P_ST_DiffDist.ipynb"),
    "E3_samsung_same": ("E3 samsung same", "modeling_10h2h_sa54_sbfe_aat_sameDist.ipynb"),
    "E4_samsung_diff": ("E4 samsung diff", "modeling_10h2h_sa54_sbfe_aat_diffDist.ipynb"),
    "E5_mixed_same": ("E5 mixed same", "modeling_6h2h_A54FEAirtags_IP12PodsST_SameDist.ipynb"),
    "E6_mixed_diff": ("E6 mixed diff", "modeling_6h2h_A54FEAirtags_IP12PodsST_DiffDist.ipynb"),
}

# Pairs used for the distance comparison (H1) and the device set comparison (H5)
DISTANCE_PAIRS = [("E1_apple_same", "E2_apple_diff"),
                  ("E3_samsung_same", "E4_samsung_diff"),
                  ("E5_mixed_same", "E6_mixed_diff")]

LABEL_MULTI = "group_label_multi"   # 0 background, 1 group 1, 2 group 2
LABEL_BINARY = "group_label"        # 0 background, 1 any target group (= group_label_bin)
LABEL_COLUMNS = ["group_label", "group_label_multi", "group_label_bin", "group_id"]

# --------------------------------------------------------------------------------------
# Feature families (Table tab:design_feature_names of the design chapter)
# --------------------------------------------------------------------------------------
FEATURE_FAMILIES = {
    "PDU role": ["is_ADV_IND", "is_ADV_NONCONN_IND", "is_ADV_SCAN_IND", "is_ADV_EXT_IND",
                 "is_AUX_ADV_IND", "is_SCAN_REQ"],
    "Channel and PHY": ["is_37", "is_38", "is_39", "is_adv_channel", "is_broadcast"],
    "Payload shape": ["has_company_id", "has_uuid16", "service_data_len_ratio"],
    "Burst microstructure": ["packets_within_2ms", "packets_within_25ms",
                             "packets_within_100ms", "packets_within_1s"],
    "Joint activity": ["bin_total_packets", "bin_distinct_sources", "coactive_packets_in_bin",
                       "coactive_sources_in_bin", "coactive_density", "is_solo_bin",
                       "bin_sync_flips"],
    "Payload dynamics": ["payload_changed", "payload_sync_in_bin", "payload_sync_density"],
    "Timing": ["iat_median", "iat_mean", "iat_std", "iat_deviation", "is_burst_iat"],
    "Shape and session span": ["len_ad_type_binned", "len_data_binned", "time_start_end_binned",
                               "delta_end_start_binned", "delta_start_start_binned",
                               "packet_send_freq_binned", "time_to_next_binned",
                               "data_len_ratio_binned", "ad_type_len_ratio_binned"],
}
FEATURE_TO_FAMILY = {f: fam for fam, feats in FEATURE_FAMILIES.items() for f in feats}


def family_of(feature: str) -> str:
    return FEATURE_TO_FAMILY.get(feature, "Other")


# --------------------------------------------------------------------------------------
# Writing (used by the modelling notebooks)
# --------------------------------------------------------------------------------------
def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")


def run_name(label: str, model_name: str, split: str, frame: str) -> str:
    """e.g. multi__random_forest__grouped__transfer"""
    lab = "multi" if label == LABEL_MULTI else "binary"
    return f"{lab}__{slug(model_name)}__{split}__{frame}"


def experiment_dir(experiment_id: str) -> pathlib.Path:
    d = ARTIFACT_DIR / experiment_id
    (d / "predictions").mkdir(parents=True, exist_ok=True)
    (d / "importance").mkdir(parents=True, exist_ok=True)
    return d


def _parquet_safe(df: pd.DataFrame) -> pd.DataFrame:
    """Object columns with mixed types (e.g. NaN and str) cannot be written to parquet."""
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == object:
            out[c] = out[c].astype("string")
    return out


def export_frames(experiment_id: str, train_df: pd.DataFrame, eval_df: pd.DataFrame) -> None:
    d = experiment_dir(experiment_id)
    _parquet_safe(train_df).to_parquet(d / "train_balanced.parquet", index=False)
    _parquet_safe(eval_df).to_parquet(d / "eval_balanced.parquet", index=False)
    print(f"exported {len(train_df)} training and {len(eval_df)} evaluation rows to {d}")


def model_scores(model, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Class scores for AUC-PR. predict_proba where available, else decision_function.

    The SGD classifier with the huber loss has no predict_proba; its decision function is
    a valid ranking score for average precision.
    """
    classes = np.asarray(model.classes_)
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X).astype(np.float32), classes
        except AttributeError:
            pass
    s = model.decision_function(X)
    if s.ndim == 1:  # binary: score of the positive class
        s = np.column_stack([-s, s])
    return s.astype(np.float32), classes


def save_predictions(experiment_id: str, name: str, rows, y_true, y_pred, scores, classes) -> pd.DataFrame:
    d = experiment_dir(experiment_id)
    pred = pd.DataFrame({"row": np.asarray(rows), "y_true": np.asarray(y_true),
                         "y_pred": np.asarray(y_pred)})
    for j, c in enumerate(classes):
        pred[f"score_{c}"] = scores[:, j]
    pred.to_parquet(d / "predictions" / f"{name}.parquet", index=False)
    return pred


def save_importance(experiment_id: str, name: str, importance_df: pd.DataFrame) -> None:
    d = experiment_dir(experiment_id)
    imp = importance_df.copy()
    imp["family"] = imp["Feature"].map(family_of)
    imp.to_csv(d / "importance" / f"{name}.csv", index=False)


def append_metrics(experiment_id: str, row: dict) -> None:
    """Upsert one run into metrics.csv (a rerun replaces the old row of the same run)."""
    d = experiment_dir(experiment_id)
    path = d / "metrics.csv"
    new = pd.DataFrame([row])
    if path.exists():
        old = pd.read_csv(path)
        old = old[old["run"] != row["run"]]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(path, index=False)


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------
def score_matrix(pred: pd.DataFrame) -> tuple[np.ndarray, list]:
    cols = [c for c in pred.columns if c.startswith("score_")]
    classes = [int(c.split("_", 1)[1]) for c in cols]
    return pred[cols].to_numpy(), classes


def auc_pr(y_true, scores: np.ndarray, classes) -> dict:
    """Average precision (area under the precision recall curve), one versus rest."""
    y_true = np.asarray(y_true)
    out = {}
    for j, c in enumerate(classes):
        pos = (y_true == c)
        out[c] = average_precision_score(pos, scores[:, j]) if 0 < pos.sum() < len(pos) else np.nan
    targets = [out[c] for c in classes if c != 0 and not np.isnan(out[c])]
    out["macro"] = float(np.nanmean([out[c] for c in classes]))
    out["targets"] = float(np.mean(targets)) if targets else np.nan
    return out


def packet_metrics(y_true, y_pred, scores=None, classes=None) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = sorted(set(np.unique(y_true)) | set(np.unique(y_pred)))
    per_class = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    m = {
        "n": len(y_true),
        "acc": accuracy_score(y_true, y_pred),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }
    for lab, f in zip(labels, per_class):
        m[f"f1_{int(lab)}"] = f
    if scores is not None:
        ap = auc_pr(y_true, scores, classes)
        m["auc_pr_macro"] = ap["macro"]
        m["auc_pr_targets"] = ap["targets"]
        for c in classes:
            m[f"auc_pr_{int(c)}"] = ap[c]
    return m


def majority_vote(meta: pd.DataFrame, y_true, y_pred, scores=None, classes=None) -> pd.DataFrame:
    """One decision per source address.

    A deployment assigns one group per address, not one group per packet. The source label
    is its (unique) true label; the prediction is the most frequent packet prediction
    (ties broken by the mean class score if scores are given, else by the smaller label).
    """
    df = pd.DataFrame({"source": np.asarray(meta["source"]),
                       "y_true": np.asarray(y_true), "y_pred": np.asarray(y_pred)})
    for col in ("labelled_device", "sublabel_device"):
        if col in meta.columns:
            df[col] = np.asarray(meta[col])
    true_src = df.groupby("source")["y_true"].agg(lambda s: s.value_counts().idxmax())
    counts = pd.crosstab(df["source"], df["y_pred"])
    if scores is not None:
        sc = pd.DataFrame(scores, columns=list(classes))
        sc["source"] = df["source"].to_numpy()
        mean_scores = sc.groupby("source").mean().reindex(index=counts.index)
        # small tie breaker: the mean score never outweighs a full vote
        tie = mean_scores.reindex(columns=counts.columns).fillna(0)
        tie = (tie.rank(axis=1) / (len(counts.columns) + 1))
        pred_src = (counts + tie * 1e-3).idxmax(axis=1)
    else:
        pred_src = counts.idxmax(axis=1)
    out = pd.DataFrame({"y_true": true_src, "y_pred": pred_src.reindex(true_src.index),
                        "n_packets": df.groupby("source").size()})
    for col in ("labelled_device", "sublabel_device"):
        if col in df.columns:
            out[col] = df.groupby("source")[col].first()
    return out.reset_index()


def source_metrics(src: pd.DataFrame) -> dict:
    return {
        "src_n": len(src),
        "src_macro_f1": f1_score(src["y_true"], src["y_pred"], average="macro", zero_division=0),
        "src_mcc": matthews_corrcoef(src["y_true"], src["y_pred"]),
        **{f"src_f1_{int(c)}": f for c, f in zip(
            sorted(src["y_true"].unique()),
            f1_score(src["y_true"], src["y_pred"], labels=sorted(src["y_true"].unique()),
                     average=None, zero_division=0))},
        # target sources only: how many of the group addresses are assigned correctly
        "src_target_recall": float((src.loc[src["y_true"] != 0, "y_true"]
                                    == src.loc[src["y_true"] != 0, "y_pred"]).mean())
        if (src["y_true"] != 0).any() else np.nan,
    }


def evaluate_run(experiment_id, label, model_name, split, frame, meta, rows, y_true, y_pred,
                 scores, classes, verbose=True) -> dict:
    """Save the predictions of one run and add its metrics row. Returns the row."""
    name = run_name(label, model_name, split, frame)
    save_predictions(experiment_id, name, rows, y_true, y_pred, scores, classes)
    m = packet_metrics(y_true, y_pred, scores, classes)
    src = majority_vote(meta, y_true, y_pred, scores, classes)
    m.update(source_metrics(src))
    row = {"experiment": experiment_id, "run": name,
           "label": "multi" if label == LABEL_MULTI else "binary",
           "model": model_name, "split": split, "frame": frame, **m}
    append_metrics(experiment_id, row)
    if verbose:
        print(f"[{frame}] macro F1 {m['macro_f1']:.3f} | MCC {m['mcc']:.3f} | "
              f"AUC-PR macro {m.get('auc_pr_macro', np.nan):.3f} | "
              f"source majority vote: macro F1 {m['src_macro_f1']:.3f}, MCC {m['src_mcc']:.3f}, "
              f"{m['src_n']} sources, target address recall {m['src_target_recall']:.3f}")
        print("Source level (majority vote) report:")
        print(classification_report(src["y_true"], src["y_pred"], zero_division=0))
    return row


# --------------------------------------------------------------------------------------
# Reading (used by the analysis notebooks)
# --------------------------------------------------------------------------------------
def available_experiments() -> list[str]:
    """Experiments whose modelling notebook has been rerun with the export (registry order first)."""
    found = [e for e in EXPERIMENTS if (ARTIFACT_DIR / e / "eval_balanced.parquet").exists()]
    if ARTIFACT_DIR.exists():
        found += sorted(d.name for d in ARTIFACT_DIR.iterdir()
                        if d.name not in EXPERIMENTS and (d / "eval_balanced.parquet").exists())
    return found


def exp_name(experiment_id: str) -> str:
    return EXPERIMENTS.get(experiment_id, (experiment_id,))[0]


def load_frame(experiment_id: str, which: str = "eval", columns=None) -> pd.DataFrame:
    """which = 'train' or 'eval'"""
    return pd.read_parquet(ARTIFACT_DIR / experiment_id / f"{which}_balanced.parquet", columns=columns)


def load_predictions(experiment_id: str, label="multi", model="random_forest", split="grouped",
                     frame="transfer", with_meta=True) -> pd.DataFrame:
    name = f"{label}__{model}__{split}__{frame}"
    pred = pd.read_parquet(ARTIFACT_DIR / experiment_id / "predictions" / f"{name}.parquet")
    if with_meta:
        meta_cols = ["source", "labelled_device", "sublabel_device", "group_id", "block", "time"]
        which = "eval" if frame == "transfer" else "train"
        meta = load_frame(experiment_id, which)
        meta = meta[[c for c in meta_cols if c in meta.columns]]
        pred = pred.join(meta, on="row")
    return pred


def load_all_metrics() -> pd.DataFrame:
    parts = [pd.read_csv(ARTIFACT_DIR / e / "metrics.csv")
             for e in available_experiments() if (ARTIFACT_DIR / e / "metrics.csv").exists()]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def feature_columns(df: pd.DataFrame) -> list[str]:
    """The 42 model features: every column that belongs to a feature family."""
    return [c for c in df.columns if c in FEATURE_TO_FAMILY]


# --------------------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------------------
def macro_f1_from_cm(cm: np.ndarray) -> np.ndarray:
    """Macro F1 from confusion matrices (..., L, L) with rows = true, columns = predicted.

    Classes that never occur (neither true nor predicted) are skipped, like sklearn does
    for labels taken from the data.
    """
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    t = cm.sum(-1)
    p = cm.sum(-2)
    denom = t + p
    f1 = np.where(denom > 0, 2 * tp / np.where(denom > 0, denom, 1), 0.0)
    present = denom > 0
    return (f1 * present).sum(-1) / np.maximum(present.sum(-1), 1)


def mcc_from_cm(cm: np.ndarray) -> np.ndarray:
    """Multi class Matthews correlation coefficient from confusion matrices (same as sklearn)."""
    t = cm.sum(-1)
    p = cm.sum(-2)
    c = np.trace(cm, axis1=-2, axis2=-1)
    s = cm.sum((-2, -1))
    num = c * s - (t * p).sum(-1)
    den = np.sqrt((s ** 2 - (p * p).sum(-1)) * (s ** 2 - (t * t).sum(-1)))
    return np.where(den > 0, num / np.where(den > 0, den, 1), 0.0)


CM_METRICS = {"macro_f1": macro_f1_from_cm, "mcc": mcc_from_cm}


def per_source_confusion(sources, y_true, y_pred, labels=None):
    """Confusion matrix of every source: array (n_sources, L, L), plus source class and ids."""
    sources = np.asarray(sources)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = np.array(sorted(set(np.unique(y_true)) | set(np.unique(y_pred)))) if labels is None else np.asarray(labels)
    li = {l: i for i, l in enumerate(labels)}
    codes, uniq = pd.factorize(sources)
    ti = np.vectorize(li.get)(y_true)
    pi = np.vectorize(li.get)(y_pred)
    L = len(labels)
    flat = np.bincount((codes * L + ti) * L + pi, minlength=len(uniq) * L * L)
    cms = flat.reshape(len(uniq), L, L).astype(np.float64)
    src_class = labels[cms.sum(-1).argmax(-1)]  # majority true class of the source
    return cms, src_class, uniq


def cluster_bootstrap(sources, y_true, preds: dict, metric="macro_f1", n_boot=1000, seed=1) -> pd.DataFrame:
    """Bootstrap over source addresses (clusters), not over packets.

    Packets of one address are strongly correlated, so resampling packets would give
    confidence intervals that are far too narrow. Sources are resampled with replacement,
    stratified by their true class so that every replicate contains target addresses.
    Uses per source confusion matrices, so 1000 replicates on 1.3 M packets take seconds.

    preds: {name: y_pred array}, all evaluated on the same rows -> paired replicates
    (the same resampled sources are used for every name).
    metric: 'macro_f1' or 'mcc'. Returns one row per replicate, one column per name.
    """
    fn = CM_METRICS[metric]
    y_true = np.asarray(y_true)
    labels = sorted(set(np.unique(y_true)).union(*[set(np.unique(p)) for p in preds.values()]))
    cms = {}
    src_class = None
    for name, p in preds.items():
        cms[name], src_class, _ = per_source_confusion(sources, y_true, p, labels)
    n_src = len(src_class)
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(src_class == s) for s in np.unique(src_class)]
    W = np.zeros((n_boot, n_src))
    for b in range(n_boot):
        for g in groups:
            W[b] += np.bincount(rng.choice(g, size=len(g), replace=True), minlength=n_src)
    out = pd.DataFrame({"replicate": np.arange(n_boot)})
    for name, cm in cms.items():
        boot_cm = np.tensordot(W, cm, axes=(1, 0))  # (n_boot, L, L)
        out[name] = fn(boot_cm)
    return out


def ci(values, alpha=0.05) -> tuple[float, float]:
    v = np.asarray(values)
    return float(np.quantile(v, alpha / 2)), float(np.quantile(v, 1 - alpha / 2))


def paired_bootstrap_test(boot: pd.DataFrame, a: str, b: str) -> dict:
    """Difference a - b with 95% CI and a two sided bootstrap p value."""
    diff = boot[a] - boot[b]
    lo, hi = ci(diff)
    p = 2 * min((diff <= 0).mean(), (diff >= 0).mean())
    return {"diff_mean": float(diff.mean()), "ci_low": lo, "ci_high": hi, "p_boot": min(1.0, float(p))}


def unpaired_bootstrap_test(a, b) -> dict:
    """Difference of two independent bootstrap distributions (different test sets,
    e.g. same against different distance, or in domain against transfer)."""
    a = np.asarray(a)
    b = np.asarray(b)
    n = min(len(a), len(b))
    diff = a[:n] - b[:n]
    lo, hi = ci(diff)
    p = 2 * min((diff <= 0).mean(), (diff >= 0).mean())
    return {"diff_mean": float(diff.mean()), "ci_low": lo, "ci_high": hi, "p_boot": min(1.0, float(p))}


def mcnemar(y_true, pred_a, pred_b) -> dict:
    """McNemar test on packet correctness of two models evaluated on the same packets.

    Note: packets of one source are not independent, so treat the p value as optimistic
    and prefer the source level bootstrap for conclusions.
    """
    from scipy.stats import chi2
    y_true = np.asarray(y_true)
    a = np.asarray(pred_a) == y_true
    b = np.asarray(pred_b) == y_true
    n01 = int((a & ~b).sum())
    n10 = int((~a & b).sum())
    stat = (abs(n01 - n10) - 1) ** 2 / (n01 + n10) if (n01 + n10) else 0.0
    return {"a_only_correct": n01, "b_only_correct": n10, "chi2": stat,
            "p_mcnemar": float(chi2.sf(stat, 1))}


def holm(pvalues) -> np.ndarray:
    """Holm-Bonferroni adjusted p values (for the many ablation comparisons)."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


# --------------------------------------------------------------------------------------
# Model used for all retraining analyses (identical to the modelling notebooks)
# --------------------------------------------------------------------------------------
def make_random_forest(n_jobs=4):
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(random_state=1, class_weight="balanced", n_estimators=50,
                                  min_samples_split=10, min_samples_leaf=3,
                                  min_weight_fraction_leaf=0.002, n_jobs=n_jobs)


def grouped_split(df: pd.DataFrame, y, test_size=0.3, grouped=True):
    """Same splits as split_data() in the modelling notebooks (seed 1)."""
    from sklearn.model_selection import GroupShuffleSplit, train_test_split
    idx = np.arange(len(df))
    if grouped:
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=1)
        tr, te = next(gss.split(idx, y, groups=df["source"]))
    else:
        tr, te = train_test_split(idx, stratify=y, test_size=test_size, random_state=1, shuffle=True)
    return tr, te
