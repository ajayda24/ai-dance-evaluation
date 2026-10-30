"""
Streamlit demo (the "Web interface" box of the architecture slide).
Run:  streamlit run app.py
"""
import tempfile
from pathlib import Path

import streamlit as st

import pipeline as P

st.set_page_config(page_title="Bharatanatyam Evaluation: Prototype", layout="wide")
st.title("AI-Based Bharatanatyam Performance Evaluation and Feedback System")
st.caption("First-review proof of concept. Body analysis is working; mudra and expression "
           "CNNs are planned (training phase).")

with st.sidebar:
    st.header("Settings")
    lang = st.selectbox("Feedback language", list(P.LANG_NAMES), format_func=lambda k: P.LANG_NAMES[k])
    target_fps = st.slider("Analysis frame rate", 6, 15, 10, help="Lower is faster")
    mirror = st.checkbox("Learner video is mirrored (webcam)", value=False)
    voice = st.checkbox("Generate voice feedback (needs internet)", value=True)
    st.markdown("---")
    st.markdown("**Pipeline**\n\n1. OpenCV frames\n2. MediaPipe landmarks\n3. Hand/face crops\n"
                "4. Joint angles\n5. DTW alignment\n6. Posture / timing / range-of-motion check\n"
                "7. Weighted score\n8. Feedback text + gTTS voice")

c1, c2 = st.columns(2)
ref_file = c1.file_uploader("Reference video (expert)", type=["mp4", "mov", "avi", "mkv"])
lrn_file = c2.file_uploader("Learner video", type=["mp4", "mov", "avi", "mkv"])


def save_tmp(up, folder):
    p = Path(folder) / up.name
    p.write_bytes(up.getbuffer())
    return str(p)


if st.button("Evaluate performance", type="primary", disabled=not (ref_file and lrn_file)):
    work = tempfile.mkdtemp(prefix="bn_")
    ref_path, lrn_path = save_tmp(ref_file, work), save_tmp(lrn_file, work)
    bar = st.progress(0.0, text="Extracting landmarks...")

    def prog(part, x):
        frac = 0.5 * min(x, 1.0) + (0.0 if part == "reference" else 0.5)
        bar.progress(frac, text=f"Extracting landmarks from {part} video...")

    try:
        res = P.run_evaluation(ref_path, lrn_path, str(Path(work) / "out"), lang, float(target_fps),
                               mirror, voice, prog)
    except Exception as exc:
        bar.empty()
        st.error(f"Could not process the videos: {exc}")
        st.stop()
    bar.empty()

    if not res["ok"]:
        st.warning(res["message"])
        st.json(res["detection"])
        st.stop()

    s = res["scores"]
    st.subheader("Score")
    m = st.columns(5)
    m[0].metric("Overall (body only for now)", f"{s['body']:.0f} / 100")
    m[1].metric("Posture", f"{s['posture']:.0f}")
    m[2].metric("Timing", f"{s['timing']:.0f}")
    m[3].metric("Range of motion", f"{s['range_of_motion']:.0f}")
    m[4].metric("Mudra / Expression", "CNN planned")
    st.caption("Overall = w1·Body + w2·Mudra + w3·Expression (late fusion). Mudra and expression models "
               "are not trained yet, so only the body term is active.")

    st.subheader("Corrective feedback")
    for line in res["feedback"]:
        st.markdown(f"- {line}")
    if res["voice_mp3"]:
        st.audio(res["voice_mp3"])
    elif voice:
        st.info("Voice skipped: gTTS needs an internet connection.")

    st.subheader("Skeleton overlay")
    v1, v2 = st.columns(2)
    v1.markdown("**Reference**")
    v1.video(res["reference_overlay"])
    v2.markdown("**Learner**")
    v2.video(res["learner_overlay"])

    st.subheader("How the comparison was made")
    st.image(res["report_png"], use_container_width=True)

    st.subheader("Landmark detection")
    d = res["detection"]
    st.dataframe({"": ["Frames analysed", "Duration (s)", "Body detected %", "Hands detected %",
                       "Face detected %"],
                  "Reference": list(d["reference"].values()),
                  "Learner": list(d["learner"].values())})

    crops = sorted(Path(res["crops_dir"]).glob("*.jpg"))
    if crops:
        st.markdown("**Hand / face regions cropped for the future mudra and expression CNNs**")
        cols = st.columns(min(len(crops), 6))
        for col, img in zip(cols, crops[:6]):
            col.image(str(img), caption=img.stem.rsplit("_", 1)[0])

    with st.expander("Per-joint numbers"):
        st.dataframe(res["joints"])
