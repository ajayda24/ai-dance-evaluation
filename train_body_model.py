"""
Train the body-error classifier (Random Forest baseline) and evaluate it the way your ML Workflow slide says:
performer-wise split (GroupKFold), accuracy, F1-score, confusion matrix. Also compares it with the
rule-based baseline used in the prototype.

  python train_body_model.py                                   # uses data/sample_body_error_dataset.csv
  python train_body_model.py --csv data/my_real_features.csv   # your real data, same columns

With the SAMPLE (synthetic) data the numbers only demonstrate the workflow. They are not results.
Outputs: outputs/body_model_rf.joblib, outputs/confusion_matrix.png, outputs/feature_importance.png
"""
import argparse
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold

import pipeline as P

LABELS = ["correct", "timing", "posture", "range_of_motion"]
META = ["clip_id", "performer_id", "dance_id", "label"]


def rule_baseline(df: pd.DataFrame) -> np.ndarray:
    """The prototype's thresholds expressed as a classifier on the same features."""
    signed = df[[c for c in df.columns if c.endswith("__dev_signed")]].abs().max(axis=1)
    rom = df[[c for c in df.columns if c.endswith("__rom_ratio")]]
    rom_bad = ((rom < P.ROM_RATIO_LOW) | (rom > P.ROM_RATIO_HIGH)).any(axis=1)
    time_bad = (df["duration_ratio"] > P.SLOWER_RATIO) | (df["duration_ratio"] < P.FASTER_RATIO)
    out = np.full(len(df), "correct", dtype=object)
    out[time_bad.to_numpy()] = "timing"
    out[rom_bad.to_numpy()] = "range_of_motion"
    out[(signed >= P.POSTURE_ERR_DEG).to_numpy()] = "posture"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/sample_body_error_dataset.csv")
    ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args()

    df = pd.read_csv(a.csv)
    feats = [c for c in df.columns if c not in META]
    X, y, groups = df[feats].fillna(0).to_numpy(), df["label"].to_numpy(), df["performer_id"].to_numpy()
    print(f"{len(df)} clips, {len(feats)} features, {len(set(groups))} performers, labels: "
          f"{dict(pd.Series(y).value_counts())}")

    rf_pred, rule_pred = np.empty(len(y), dtype=object), rule_baseline(df)
    fold_rows = []
    for k, (tr, te) in enumerate(GroupKFold(n_splits=a.folds).split(X, y, groups), 1):
        clf = RandomForestClassifier(n_estimators=300, random_state=k, n_jobs=-1)
        clf.fit(X[tr], y[tr])
        rf_pred[te] = clf.predict(X[te])
        fold_rows.append({"fold": k, "test_performers": ",".join(sorted(set(groups[te]))),
                          "RF accuracy": accuracy_score(y[te], rf_pred[te]),
                          "RF macro-F1": f1_score(y[te], rf_pred[te], average="macro"),
                          "Rule accuracy": accuracy_score(y[te], rule_pred[te]),
                          "Rule macro-F1": f1_score(y[te], rule_pred[te], average="macro")})
    folds = pd.DataFrame(fold_rows)
    print("\nPer fold (each fold tests on performers never seen in training):")
    print(folds.round(3).to_string(index=False))

    print("\nOverall (all folds pooled):")
    for name, pred in (("Rule-based baseline", rule_pred), ("Random Forest", rf_pred)):
        print(f"  {name:20s} accuracy {accuracy_score(y, pred):.3f}   macro-F1 {f1_score(y, pred, average='macro'):.3f}")

    cm = confusion_matrix(y, rf_pred, labels=LABELS)
    print("\nRandom Forest confusion matrix (rows = true, columns = predicted):")
    print(pd.DataFrame(cm, index=LABELS, columns=LABELS).to_string())

    out = Path("outputs")
    out.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.8, 5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(4)); ax.set_xticklabels(LABELS, rotation=30, ha="right")
    ax.set_yticks(range(4)); ax.set_yticklabels(LABELS)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Random Forest, performer-wise CV\n(sample data: workflow demo)")
    fig.tight_layout(); fig.savefig(out / "confusion_matrix.png", dpi=140); plt.close(fig)

    final = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1).fit(X, y)
    joblib.dump({"model": final, "features": feats, "labels": list(final.classes_)}, out / "body_model_rf.joblib")
    imp = pd.Series(final.feature_importances_, index=feats).sort_values()[-12:]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh([i.replace("__", " ").replace("_", " ") for i in imp.index], imp.values, color="#2d3770")
    ax.set_title("Top features (Random Forest)"); ax.set_xlabel("importance")
    fig.tight_layout(); fig.savefig(out / "feature_importance.png", dpi=140); plt.close(fig)
    print("\nSaved outputs/body_model_rf.joblib, confusion_matrix.png, feature_importance.png")


if __name__ == "__main__":
    main()
