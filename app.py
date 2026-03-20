import streamlit as st
st.set_page_config(page_title="Face Recognition — Streamlit", layout="wide")

import os
from PIL import Image, ImageDraw
import numpy as np
from engine import FaceEngine
from database import FaceDatabase
import io

# ---------- Load CSS ----------
if os.path.exists("ui.css"):
    with open("ui.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ---------- Config ----------
THRESHOLD = 0.30
DB_FOLDER = "faces"

# ---------- Helpers ----------
@st.cache_resource(show_spinner=False)
def get_engine():
    return FaceEngine(ctx_id=-1, model_name="buffalo_l")

@st.cache_resource(show_spinner=False)
def get_db():
    return FaceDatabase(engine=get_engine(), db_path=DB_FOLDER)

def normalize_emb(v):
    v = np.asarray(v, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-9)

def draw_bbox_and_label(img_bgr, face_obj, label):
    img_rgb = img_bgr[:, :, ::-1]
    img_pil = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(img_pil)

    x1, y1, x2, y2 = map(int, face_obj.bbox)
    draw.rectangle([x1, y1, x2, y2], outline="green", width=3)
    draw.text((x1, y1 - 10), label, fill="green")

    return np.array(img_pil)[:, :, ::-1]

# ---------- Init ----------
engine = get_engine()
db = get_db()

# ---------- UI ----------
st.title("🔵 Face Recognition System")
st.caption("Upload an image or use your webcam. (Powered by InsightFace)")

with st.sidebar:
    st.header("Controls")
    mode = st.radio("Mode", ["Recognize (Image)", "Live Webcam", "Add Person", "Manage Dataset"])
    st.markdown("---")
    thresh = st.slider("Similarity threshold", 0.0, 1.0, THRESHOLD, 0.01)
    st.text("Model: InsightFace (buffalo_l)")

# ---------- MODE: Recognize ----------
if mode == "Recognize (Image)":
    st.subheader("🔍 Recognize from uploaded image")
    uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        bgr = np.array(img)[:, :, ::-1]

        with st.spinner("Detecting face..."):
            face_obj, _ = engine.best_face(bgr)

        if face_obj is None:
            st.error("No face detected.")
        else:
            emb = engine.get_embedding_from_faceobj(face_obj)

            if emb is None:
                st.error("Embedding failed.")
            else:
                emb = normalize_emb(emb)
                embeddings = db.load_all_embeddings()

                if not embeddings:
                    st.info("No people in database.")
                else:
                    best_name = "Unknown"
                    best_score = -1

                    for name, saved in embeddings.items():
                        score = float(np.dot(emb, saved))
                        if score > best_score:
                            best_score = score
                            best_name = name

                    if best_score < thresh:
                        best_name = "Unknown"

                    bgr = draw_bbox_and_label(bgr, face_obj, f"{best_name} {best_score:.2f}")

                    st.image(
                        bgr[:, :, ::-1],
                        use_column_width=True,
                        caption=f"{best_name} ({best_score:.2f})"
                    )

# ---------- MODE: Webcam ----------
elif mode == "Live Webcam":
    st.subheader("📷 Webcam Snapshot")
    snapshot = st.camera_input("Take a photo")

    if snapshot:
        img = Image.open(io.BytesIO(snapshot.getvalue())).convert("RGB")
        bgr = np.array(img)[:, :, ::-1]

        face_obj, _ = engine.best_face(bgr)

        if face_obj is None:
            st.error("No face detected.")
        else:
            emb = normalize_emb(engine.get_embedding_from_faceobj(face_obj))
            embeddings = db.load_all_embeddings()

            best_name = "Unknown"
            best_score = -1

            for name, saved in embeddings.items():
                score = float(np.dot(emb, saved))
                if score > best_score:
                    best_score = score
                    best_name = name

            if best_score < thresh:
                best_name = "Unknown"

            bgr = draw_bbox_and_label(bgr, face_obj, f"{best_name} {best_score:.2f}")

            st.image(bgr[:, :, ::-1], caption=f"{best_name} ({best_score:.2f})")

# ---------- MODE: Add Person ----------
elif mode == "Add Person":
    st.subheader("➕ Add Person")
    name = st.text_input("Enter name (no spaces)")

    uploads = st.file_uploader("Upload 3 images", accept_multiple_files=True)

    if st.button("Save Person"):
        if not name:
            st.error("Enter name")
        elif not uploads or len(uploads) != 3:
            st.error("Upload exactly 3 images")
        else:
            embs = []
            for f in uploads:
                img = Image.open(f).convert("RGB")
                bgr = np.array(img)[:, :, ::-1]

                face_obj, _ = engine.best_face(bgr)
                if face_obj is None:
                    st.error("Face not detected in one image")
                    break

                emb = engine.get_embedding_from_faceobj(face_obj)
                embs.append(normalize_emb(emb))

            if len(embs) == 3:
                db.add_person(name, embs)
                st.success(f"{name} added!")

# ---------- MODE: Manage ----------
elif mode == "Manage Dataset":
    st.subheader("🗂 Dataset")

    people = db.list_people()

    if not people:
        st.info("No data")
    else:
        for p in people:
            st.write(p)

        delete = st.selectbox("Delete", ["Select"] + people)

        if delete != "Select" and st.button("Delete"):
            db.delete_person(delete)
            st.success(f"{delete} deleted")

# ---------- Footer ----------
st.markdown("---")
st.caption("Add people first, then test recognition.")
