"""
Turn your REAL labelled clips into the same feature table as the sample dataset.

1. Put reference videos and learner clips in folders, fill in two CSV files:
     data/references.csv   dance_id,video_path                     (one expert video per dance)
     data/labels.csv       clip_id,performer_id,dance_id,file_path,error_label   (see labels_template.csv)
   error_label must be one of: correct, timing, posture, range_of_motion
2. Run:   python extract_dataset.py
3. Train: python train_body_model.py --csv data/real_body_error_dataset.csv
Clips where the body cannot be tracked are skipped and listed at the end (re-record those).
"""
import argparse
from pathlib import Path

import pandas as pd

import pipeline as P

VALID = {"correct", "timing", "posture", "range_of_motion"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/labels.csv")
    ap.add_argument("--references", default="data/references.csv")
    ap.add_argument("--out", default="data/real_body_error_dataset.csv")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--mirror", action="store_true", help="clips were recorded through a mirrored webcam")
    a = ap.parse_args()

    labels = pd.read_csv(a.labels)
    refs = pd.read_csv(a.references).set_index("dance_id")["video_path"].to_dict()
    ref_cache, rows, skipped = {}, [], []

    for _, r in labels.iterrows():
        if r["error_label"] not in VALID:
            skipped.append((r["clip_id"], f"bad label '{r['error_label']}'"))
            continue
        try:
            if r["dance_id"] not in ref_cache:
                f = P.extract_features(refs[r["dance_id"]], a.fps)
                ref_cache[r["dance_id"]] = (P.joint_angles(f), f.fps)
            R, fps_r = ref_cache[r["dance_id"]]
            f = P.extract_features(r["file_path"], a.fps)
            L = P.joint_angles(f)
            if a.mirror:
                L = P.mirror_angles(L)
            res = P.evaluate_body(L, R, f.fps, fps_r)
        except Exception as exc:
            skipped.append((r["clip_id"], str(exc)))
            continue
        row = {"clip_id": r["clip_id"], "performer_id": r["performer_id"],
               "dance_id": r["dance_id"], "label": r["error_label"]}
        row.update(P.feature_row(res))
        rows.append(row)
        print(f"clip {r['clip_id']}: {r['error_label']}")

    if rows:
        pd.DataFrame(rows).to_csv(a.out, index=False)
        print(f"\nSaved {len(rows)} clips -> {a.out}")
    if skipped:
        print("\nSkipped:")
        for cid, why in skipped:
            print(f"  clip {cid}: {why}")


if __name__ == "__main__":
    main()
