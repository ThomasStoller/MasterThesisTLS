"""One-off patch: extends the six experiment modelling notebooks for the final rerun.

Changes per notebook (idempotent, safe to run twice):
  1. CODE FIELD 1   : EXPERIMENT_ID variable and a 'block' column (0..4 training, -1 university).
  2. dropped_columns: 'block' is metadata, never a feature.
  3. CODE FIELD 3b  : new cell, exports the balanced train / eval frames to parquet.
  4. model_feature_importance() returns its table so it can be saved.
  5. CODE FIELD 6   : runs BOTH labels (3 class and binary), saves per-packet predictions
                      and class scores, feature importance, and a metrics row with
                      macro F1, MCC, AUC-PR and the per-source majority vote.
                      The printed output keeps the old format, so results_comparison.ipynb
                      still parses it.

Usage (from the repository root):
    python code/thesis/analysis/patch_modeling_notebooks.py
"""
import json
import pathlib
import sys

MODELING = pathlib.Path(__file__).resolve().parents[1] / "modeling"

NOTEBOOKS = {
    "E1_apple_same": "modeling_10h2h_IP12_AAP1P_ST_SameDist.ipynb",
    "E2_apple_diff": "modeling_10h2h_IP12_AAP1P_ST_DiffDist.ipynb",
    "E3_samsung_same": "modeling_10h2h_sa54_sbfe_aat_sameDist.ipynb",
    "E4_samsung_diff": "modeling_10h2h_sa54_sbfe_aat_diffDist.ipynb",
    "E5_mixed_same": "modeling_6h2h_A54FEAirtags_IP12PodsST_SameDist.ipynb",
    "E6_mixed_diff": "modeling_6h2h_A54FEAirtags_IP12PodsST_DiffDist.ipynb",
}

EXPORT_CELL = """##############################
#  CODE FIELD 3b - Export    #
##############################

# Export the two balanced frames once. All analysis notebooks in code/thesis/analysis read these
# files, so the ablation, significance tests, error analysis and clustering never need the
# feature engineering above again. Written to code/thesis/modeling/artifacts/<EXPERIMENT_ID>/.
au.export_frames(EXPERIMENT_ID, train_df_binned_large_balanced, eval_df_binned_balanced)
"""

FIELD6_CELL = """##############################
#        CODE FIELD 6        #
##############################

# Cross-location setup: train = Seewis capture (EXPERIMENT_ID), eval = 2h uni held-out location.
# GroupShuffleSplit below additionally guards against same-source (RPA) leakage within train.
#
# Every run is evaluated twice: 'indomain' = held-out 30 % of the training capture,
# 'transfer' = the university capture. Both labels are run in one go:
#   group_label_multi : 0 background, 1 group 1, 2 group 2 (main task)
#   group_label       : 0 background, 1 any target group (binary baseline of the earlier theses, H2)
# For every run the per-packet predictions and class scores, the feature importance and one
# metrics row (macro F1, MCC, AUC-PR, per-source majority vote) are written to
# code/thesis/modeling/artifacts/<EXPERIMENT_ID>/ by analysis_utils.

# Split-mode False = stratified packet shuffle (leakage prone), True = GroupShuffleSplit by source (recommended for RPA)
split_mode_grouped = [False, True]

# Labels to run. Remove one to save time if only one task is needed.
label_col_binary = 'group_label'
label_col_multi = 'group_label_multi'
labels_to_run = [label_col_multi, label_col_binary]

df_to_train = train_df_binned_large_balanced
df_to_eval = eval_df_binned_balanced

# Make sure column order matches (important for single-file duplicated data)
if (df_to_train.columns != df_to_eval.columns).any():
    df_to_eval = df_to_eval[df_to_train.columns]

feature_drop = [c for c in dropped_columns + [label_col_binary, label_col_multi, 'group_id', 'group_label_bin'] if c in df_to_train.columns]
X_all = df_to_train.drop(columns=feature_drop)
X_eval = df_to_eval.drop(columns=feature_drop)
meta_cols = ['source', 'labelled_device', 'sublabel_device']
print(f'USING COLUMNS ({X_all.shape[1]}): {list(X_all.columns)}')

for use_label in labels_to_run:
    print(f'\\n############ LABEL {use_label} ############')
    print(f'Training distribution: {df_to_train[use_label].value_counts().to_dict()}')
    print(f'Eval uni distribution: {df_to_eval[use_label].value_counts().to_dict()} (held-out location)')

    for model_name in model_dict.keys():
        for split_mode in split_mode_grouped:
            model_to_run = model_dict[model_name]
            split_name = 'grouped' if split_mode else 'packet'

            print(f'\\n------------------------------------------------------------')
            print(f'STARTING MODEL: {model_name} WITH GROUPED SPLIT MODE: {split_mode} AND TEST SIZE: {test_param} LABEL: {use_label}')
            print(f'------------------------------------------------------------\\n')

            X_train, X_test, y_train, y_test = split_data(
                base_df = df_to_train,
                X = X_all,
                y = df_to_train[use_label],
                test_size = test_param,
                split_mode_group=split_mode
            )

            model, y_pred = fit_and_predict_model(
                model = model_to_run,
                X_train = X_train,
                X_test = X_test,
                y_train = y_train,
                y_test = y_test
            )

            # ---- in domain (held-out part of the training capture) ----
            print(f'Model {model_name}', classification_report(y_test, y_pred, zero_division=0))
            create_matrix(model=model, y_test=y_test, y_pred=y_pred, model_name=model_name,
                          split_mode=split_mode, model_type='Training')
            scores, classes = au.model_scores(model, X_test)
            au.evaluate_run(EXPERIMENT_ID, use_label, model_name, split_name, 'indomain',
                            meta=df_to_train.loc[X_test.index, meta_cols], rows=X_test.index,
                            y_true=y_test, y_pred=y_pred, scores=scores, classes=classes)

            # Importance is computed once per run (the old second 'Evaluation' call computed the same thing again)
            importance_df = model_feature_importance(model=model, X_test=X_test, y_test=y_test,
                                                     model_name=model_name, model_type='Training')
            au.save_importance(EXPERIMENT_ID, au.run_name(use_label, model_name, split_name, 'indomain'), importance_df)

            # ---- transfer (university capture, held-out location) ----
            y_eval_true = df_to_eval[use_label]
            y_eval_pred = model.predict(X_eval)
            print(f'Evaluation on Test Dataset (label={use_label}):')
            print(classification_report(y_eval_true, y_eval_pred, zero_division=0))
            create_matrix(model=model, y_test=y_eval_true, y_pred=y_eval_pred, model_name=model_name,
                          split_mode=split_mode, model_type='Evaluation')
            scores, classes = au.model_scores(model, X_eval)
            au.evaluate_run(EXPERIMENT_ID, use_label, model_name, split_name, 'transfer',
                            meta=df_to_eval[meta_cols], rows=np.arange(len(df_to_eval)),
                            y_true=y_eval_true, y_pred=y_eval_pred, scores=scores, classes=classes)

            del X_train, X_test, y_train, y_test, y_pred, y_eval_pred, scores
            gc.collect()

print('\\nAll runs written to', au.experiment_dir(EXPERIMENT_ID))
"""

IMPORT_SNIPPET = """
# Shared helpers for exporting predictions and metrics (code/thesis/analysis/analysis_utils.py)
import sys
sys.path.append('../analysis')
import importlib
import analysis_utils as au
importlib.reload(au)
"""


def src(cell):
    return "".join(cell["source"])


def set_src(cell, text):
    lines = text.splitlines(keepends=True)
    cell["source"] = lines


def find(cells, marker):
    hits = [i for i, c in enumerate(cells) if marker in src(c)]
    if len(hits) != 1:
        raise RuntimeError(f"marker {marker!r} found {len(hits)} times")
    return hits[0]


def new_code_cell(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": text.splitlines(keepends=True)}


def patch(exp_id, fname):
    path = MODELING / fname
    nb = json.loads(path.read_text(encoding="utf-8"))
    cells = nb["cells"]
    changed = []

    # 0) imports
    i = find(cells, "from sklearn.ensemble import RandomForestClassifier")
    if "analysis_utils" not in src(cells[i]):
        set_src(cells[i], src(cells[i]).rstrip("\n") + "\n" + IMPORT_SNIPPET)
        changed.append("imports")

    # 1) CODE FIELD 1: experiment id and block column
    i = find(cells, "#        CODE FIELD 1        #")
    s = src(cells[i])
    if "EXPERIMENT_ID" not in s:
        s = s.replace("#        CODE FIELD 1        #\n##############################\n",
                      "#        CODE FIELD 1        #\n##############################\n\n"
                      f"# Name of the artifact folder in code/thesis/modeling/artifacts\nEXPERIMENT_ID = '{exp_id}'\n", 1)
        assert s.count("for f in train_files:") == 1
        s = s.replace("for f in train_files:", "for block_idx, f in enumerate(train_files):")
        assert s.count("    ic = combine_data(raw, target_list)\n") == 1
        s = s.replace("    ic = combine_data(raw, target_list)\n",
                      "    ic = combine_data(raw, target_list)\n"
                      "    ic['block'] = block_idx  # two hour block index, used for per block analyses\n")
        assert s.count("int_eval = combine_data(eval_raw, target_list)\n") == 1
        s = s.replace("int_eval = combine_data(eval_raw, target_list)\n",
                      "int_eval = combine_data(eval_raw, target_list)\n"
                      "int_eval['block'] = -1  # university capture\n")
        set_src(cells[i], s)
        changed.append("field1")

    # 2) dropped columns
    i = find(cells, "dropped_columns = [")
    s = src(cells[i])
    if "'block'," not in s:
        assert s.count("    'group_label_multi',\n") == 1
        s = s.replace("    'group_label_multi',\n",
                      "    'group_label_multi',\n    'block',  # metadata: two hour block index\n")
        set_src(cells[i], s)
        changed.append("dropped")

    # 3) export cell after CODE FIELD 3
    if not any("CODE FIELD 3b - Export" in src(c) for c in cells):
        i = find(cells, "#        CODE FIELD 3        #")
        cells.insert(i + 1, new_code_cell(EXPORT_CELL))
        changed.append("export")

    # 4) model_feature_importance returns its table
    i = find(cells, "def model_feature_importance(")
    s = src(cells[i])
    if "return importance_df" not in s:
        set_src(cells[i], s.rstrip("\n") + "\n\n    return importance_df\n")
        changed.append("importance")

    # 5) CODE FIELD 6
    i = find(cells, "#        CODE FIELD 6        #")
    if "labels_to_run" not in src(cells[i]):
        set_src(cells[i], FIELD6_CELL)
        cells[i]["outputs"] = []
        cells[i]["execution_count"] = None
        changed.append("field6")

    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{exp_id:16s} {fname}: {', '.join(changed) or 'already patched'}")


if __name__ == "__main__":
    for exp_id, fname in NOTEBOOKS.items():
        patch(exp_id, fname)
    sys.exit(0)
