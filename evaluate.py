"""
Command-line evaluation (no browser needed).

  python evaluate.py reference.mp4 learner.mp4 --lang ml --out outputs/run1

Outputs in the --out folder: report.png, learner_overlay.mp4, reference_overlay.mp4,
crops_learner/ (hand + face crops), feedback.mp3 (needs internet) and result.json.
"""
import argparse
import sys

import pipeline as P


def main():
    ap = argparse.ArgumentParser(description="Compare a learner video with a reference video.")
    ap.add_argument("reference")
    ap.add_argument("learner")
    ap.add_argument("--lang", default="en", choices=list(P.LANG_NAMES))
    ap.add_argument("--out", default="outputs/run1")
    ap.add_argument("--fps", type=float, default=10.0, help="analysis frame rate (default 10)")
    ap.add_argument("--mirror", action="store_true", help="learner video is mirrored (typical webcam)")
    ap.add_argument("--no-voice", action="store_true", help="skip gTTS (offline)")
    a = ap.parse_args()

    res = P.run_evaluation(a.reference, a.learner, a.out, a.lang, a.fps, a.mirror,
                           voice=not a.no_voice,
                           progress=lambda part, x: print(f"\r{part}: {x*100:3.0f}%", end="", flush=True))
    print()
    if not res["ok"]:
        print(res["message"])
        sys.exit(1)
    print("Scores:", res["scores"])
    print("Detection:", res["detection"])
    print("\nFeedback:")
    for line in res["feedback"]:
        print(" -", line)
    print("\nSaved to:", a.out)


if __name__ == "__main__":
    main()
