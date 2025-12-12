# engine.py
import cv2
import numpy as np
from insightface.app import FaceAnalysis

class FaceEngine:
    """
    Single engine that runs detection + alignment + embedding via InsightFace.
    Uses FaceAnalysis (buffalo_l) which contains SCRFD detector + ArcFace-style embedding.
    """

    def __init__(self, ctx_id=-1, model_name="buffalo_l"):
        """
        ctx_id: -1 -> CPU, 0 -> first GPU (change if you have GPU and onnxruntime-gpu)
        """
        print("[engine] loading model:", model_name, " ctx_id=", ctx_id)
        self.app = FaceAnalysis(name=model_name)
        self.app.prepare(ctx_id=ctx_id)   # -1 for CPU; 0 for GPU

    def get_faces(self, bgr_image):
        """
        Returns list of InsightFace face objects for the image (bgr input).
        Each face object has .bbox (4), .kps (5 landmarks), .embedding (512-d float array).
        """
        # FaceAnalysis expects BGR or RGB? it handles internally; we'll pass BGR directly.
        faces = self.app.get(bgr_image)
        return faces or []

    def best_face(self, bgr_image):
        """
        Return the single best face (largest area) or None.
        Also return the cropped face image (BGR).
        """
        faces = self.get_faces(bgr_image)
        if not faces:
            return None, None

        # pick the largest bounding box area
        best = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
        x1, y1, x2, y2 = best.bbox.astype(int)
        # clamp
        h, w = bgr_image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = bgr_image[y1:y2, x1:x2].copy()
        return best, crop

    def get_embedding_from_faceobj(self, face_obj):
        """
        face_obj is an InsightFace face object returned by app.get(...)
        It typically already contains .embedding (512-d). Return normalized vector.
        """
        if face_obj is None:
            return None
        emb = getattr(face_obj, "embedding", None)
        if emb is None:
            return None
        emb = np.array(emb, dtype=np.float32)
        # L2 normalize
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb

    def get_embedding_from_crop(self, crop_bgr):
        """
        Convenience: run detection on crop and return embedding (if any).
        """
        faces = self.get_faces(crop_bgr)
        if not faces:
            return None
        return self.get_embedding_from_faceobj(faces[0])
