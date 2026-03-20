import streamlit as st
st.set_page_config(page_title="Face Recognition", layout="wide")

from PIL import Image, ImageDraw
import numpy as np
import io
import os

from engine import FaceEngine
from database import FaceDatabase

# ---------- Config ----------
THRESHOLD = 0.5
DB_FOLDER = "faces"

# ---------- Load CSS ----------
if os.path.exists("ui.css"):
    with open("ui.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ---------- Init ----------
engine = FaceEngine()
db = FaceDatabase(DB_FOLDER)

# ---------- Helpers ----------
def draw_bbox_and_label(img_rgb, box, label):
    img_pil = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(img_pil)

    top, right, bottom, left = box
    draw.rectangle([left, top, right, bottom], outline="green", width=3)
    draw.text((left, top - 10), label, fill="green")

    return np.array(img_pil)

# ---------- UI ----------
st.title("🔵 Face Recognition System")
st.caption("Lightweight version using face_recognition")

mode = st.sidebar.radio(
    "Mode",
    ["Recognize", "Webcam", "Add Person", "Manage Dataset"]
)

# ---------- Recognize ----------
if mode == "Recognize":
    uploaded = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        rgb = np.array(img)

        box, emb = engine.best_face(rgb)

        if box is None:
            st.error("No face detected")
        else:
            embeddings = db.load_all_embeddings()

            best_name = "Unknown"
            best_score = 0

            for name, saved in embeddings.items():
                dist = np.linalg.norm(emb - saved)
                score = 1 - dist

                if score > best_score:
                    best_score = score
                    best_name = name

            if best_score < THRESHOLD:
                best_name = "Unknown"

            rgb = draw_bbox_and_label(rgb, box, f"{best_name} {best_score:.2f}")
            st.image(rgb, caption=f"{best_name} ({best_score:.2f})")

# ---------- Webcam ----------
elif mode == "Webcam":
    snap = st.camera_input("Take Photo")

    if snap:
        img = Image.open(io.BytesIO(snap.getvalue())).convert("RGB")
        rgb = np.array(img)

        box, emb = engine.best_face(rgb)

        if box is None:
            st.error("No face detected")
        else:
            embeddings = db.load_all_embeddings()

            best_name = "Unknown"
            best_score = 0

            for name, saved in embeddings.items():
                dist = np.linalg.norm(emb - saved)
                score = 1 - dist

                if score > best_score:
                    best_score = score
                    best_name = name

            if best_score < THRESHOLD:
                best_name = "Unknown"

            rgb = draw_bbox_and_label(rgb, box, f"{best_name} {best_score:.2f}")
            st.image(rgb)

# ---------- Add Person ----------
elif mode == "Add Person":
    name = st.text_input("Enter Name")

    uploads = st.file_uploader("Upload 3 images", accept_multiple_files=True)

    if st.button("Save"):
        if not name or len(uploads) != 3:
            st.error("Provide name and 3 images")
        else:
            embs = []

            for f in uploads:
                img = Image.open(f).convert("RGB")
                rgb = np.array(img)

                box, emb = engine.best_face(rgb)
                if box is None:
                    st.error("Face not detected in one image")
                    break

                embs.append(emb)

            if len(embs) == 3:
                db.add_person(name, embs)
                st.success("Person added!")

# ---------- Manage ----------
elif mode == "Manage Dataset":
    people = db.list_people()

    if not people:
        st.info("No data")
    else:
        for p in people:
            st.write(p)

        delete = st.selectbox("Delete", ["Select"] + people)

        if delete != "Select" and st.button("Delete"):
            db.delete_person(delete)
            st.success("Deleted")

# ---------- Footer ----------
st.markdown("---")
st.caption("Add people first, then test recognition.")
