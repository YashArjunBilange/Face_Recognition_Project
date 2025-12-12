import streamlit as st
from PIL import Image
import numpy as np
import cv2
from engine import FaceEngine
from database import FaceDatabase
import os
import io
import time

# at top of app.py after imports
if os.path.exists("ui.css"):
    with open("ui.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ---------- Config ----------
THRESHOLD = 0.30
DB_FOLDER = "faces"
st.set_page_config(page_title="Face Recognition — Streamlit", layout="wide")

# ---------- Helpers & Cached resources ----------
@st.cache_resource(show_spinner=False)
def get_engine():
    # load a CPU-friendly model
    engine = FaceEngine(ctx_id=-1, model_name="antelopev2")
    return engine

@st.cache_resource(show_spinner=False)
def get_db():
    return FaceDatabase(engine=get_engine(), db_path=DB_FOLDER)

def normalize_emb(v):
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v) + 1e-9
    return v / n

def draw_bbox_and_label(img_bgr, face_obj, label):
    x1, y1, x2, y2 = face_obj.bbox.astype(int)
    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 200, 0), 3)
    cv2.putText(img_bgr, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)

# ---------- Init ----------
engine = get_engine()
db = get_db()

# Top UI
st.markdown("<style> .stApp { background-color: #0b0f13;} </style>", unsafe_allow_html=True)
st.title("🔵 Face Recognition System")
st.caption("Upload an image or use your webcam. (Powered by InsightFace)")

# Layout: sidebar for controls, main for actions
with st.sidebar:
    st.header("Controls")
    mode = st.radio("Mode", ["Recognize (Image)", "Live Webcam", "Add Person", "Manage Dataset"])
    st.markdown("---")
    st.write("Threshold (cosine similarity)")
    thresh = st.slider("Similarity threshold", 0.0, 1.0, THRESHOLD, 0.01)
    st.write("Model:")
    st.text("antelopev2 (CPU)")

# ---------- MODE: Recognize (Image) ----------
if mode == "Recognize (Image)":
    st.subheader("🔍 Recognize from uploaded image")
    uploaded = st.file_uploader("Upload one image", type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        img = Image.open(uploaded).convert("RGB")
        bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        with st.spinner("Detecting face..."):
            face_obj, crop = engine.best_face(bgr)

        if face_obj is None:
            st.error("No face detected in the image.")
        else:
            emb = engine.get_embedding_from_faceobj(face_obj)
            if emb is None:
                st.error("Failed to get embedding for the detected face.")
            else:
                emb = normalize_emb(emb)
                embeddings = db.load_all_embeddings()
                if not embeddings:
                    st.info("No people in the database. Add persons first.")
                else:
                    best_name = "Unknown"
                    best_score = -1.0
                    for name, saved in embeddings.items():
                        score = float(np.dot(emb, saved))
                        if score > best_score:
                            best_score = score
                            best_name = name
                    if best_score < thresh:
                        best_name = "Unknown"

                    # draw bbox + label
                    draw_bbox_and_label(bgr, face_obj, f"{best_name} {best_score:.2f}")
                    st.image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), use_column_width=True,
                             caption=f"Prediction: {best_name} (score={best_score:.2f})")

# ---------- MODE: Live Webcam ----------
elif mode == "Live Webcam":
    st.subheader("📷 Live Webcam (browser capture)")
    st.info("Click 'Take snapshot' to capture from your webcam. This sends one image to the server for recognition.")
    snapshot = st.camera_input("Point your camera at your face and click 'Take snapshot'")

    if snapshot is not None:
        bytes_data = snapshot.getvalue()
        img = Image.open(io.BytesIO(bytes_data)).convert("RGB")
        bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        with st.spinner("Detecting face..."):
            face_obj, crop = engine.best_face(bgr)

        if face_obj is None:
            st.error("No face detected. Try again with good lighting and a single face in frame.")
        else:
            emb = engine.get_embedding_from_faceobj(face_obj)
            if emb is None:
                st.error("Failed to get embedding.")
            else:
                emb = normalize_emb(emb)
                embeddings = db.load_all_embeddings()
                if not embeddings:
                    st.info("No people in database yet.")
                else:
                    best_name = "Unknown"
                    best_score = -1.0
                    for name, saved in embeddings.items():
                        score = float(np.dot(emb, saved))
                        if score > best_score:
                            best_score = score
                            best_name = name
                    if best_score < thresh:
                        best_name = "Unknown"

                    draw_bbox_and_label(bgr, face_obj, f"{best_name} {best_score:.2f}")
                    st.image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), caption=f"{best_name} ({best_score:.2f})")

# ---------- MODE: Add Person ----------
elif mode == "Add Person":
    st.subheader("➕ Register a new person")
    st.markdown("Upload **exactly 3** images (front, left, right) OR use the webcam three times.")
    name = st.text_input("Enter person's name (no spaces):")
    col1, col2 = st.columns(2)

    with col1:
        uploads = st.file_uploader("Upload 3 images (recommended)", type=["jpg", "png", "jpeg"],
                                   accept_multiple_files=True, key="uploads")
    with col2:
        st.write("OR use webcam:")
        snap1 = st.camera_input("Pose: Front", key="snap1")
        snap2 = st.camera_input("Pose: Left", key="snap2")
        snap3 = st.camera_input("Pose: Right", key="snap3")

    if st.button("Save Person"):
        imgs = []
        # prefer uploads if provided
        if uploads and len(uploads) == 3:
            for f in uploads:
                im = Image.open(f).convert("RGB")
                imgs.append(cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR))
        else:
            # collect from snapshots if available
            snaps = [snap1, snap2, snap3]
            for s in snaps:
                if s is not None:
                    im = Image.open(io.BytesIO(s.getvalue())).convert("RGB")
                    imgs.append(cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR))

        if not name:
            st.error("Please enter a name.")
        elif len(imgs) != 3:
            st.error("Please provide exactly 3 images (either upload 3 files or take 3 snapshots).")
        else:
            embs = []
            failed = False
            with st.spinner("Processing images and extracting embeddings..."):
                for i, frame in enumerate(imgs):
                    face_obj, crop = engine.best_face(frame)
                    if face_obj is None:
                        st.warning(f"No face detected in image #{i+1}. Operation aborted.")
                        failed = True
                        break
                    emb = engine.get_embedding_from_faceobj(face_obj)
                    if emb is None:
                        st.warning(f"Failed to compute embedding for image #{i+1}.")
                        failed = True
                        break
                    embs.append(normalize_emb(emb))

            if not failed:
                db.add_person(name, embs)
                # clear cache for db listing
                try:
                    # invalidate list cache if necessary
                    st.cache_resource.clear()
                except Exception:
                    pass
                st.success(f"Added person '{name}' with {len(embs)} embeddings.")

# ---------- MODE: Manage Dataset ----------
elif mode == "Manage Dataset":
    st.subheader("🗂 Manage Dataset")
    people = db.list_people()
    if not people:
        st.info("No people saved yet.")
    else:
        cols = st.columns((3, 1))
        with cols[0]:
            for p in people:
                with st.expander(p):
                    # show saved embedding preview (just the name & np shape)
                    emb = np.load(db.person_file(p))
                    st.write("Embedding shape:", emb.shape)
                    st.write("Name:", p)
        with cols[1]:
            to_delete = st.selectbox("Delete person", ["Select"] + people)
            if to_delete != "Select":
                if st.button("Delete selected"):
                    db.delete_person(to_delete)
                    # clear cache to refresh listing
                    try:
                        st.cache_resource.clear()
                    except Exception:
                        pass
                    st.success(f"Deleted {to_delete}")

# Footer
st.markdown("---")
st.caption("Tip: first add persons using 'Add Person' then use 'Live Webcam' or 'Recognize (Image)' to test.")
