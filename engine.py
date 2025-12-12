import cv2
import numpy as np
from insightface.app import FaceAnalysis

class FaceEngine:
    def __init__(self, ctx_id=-1, model_name="buffalo_l"):
        # Use a lightweight CPU-friendly model and force CPU provider
        # FaceAnalysis will download model files on first run
        self.app = FaceAnalysis(name=model_name, providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=ctx_id)

    def get_faces(self, bgr):
        # FaceAnalysis expects BGR images
        return self.app.get(bgr)

    def best_face(self, bgr):
        faces = self.get_faces(bgr)
        if not faces:
            return None, None
        # choose the largest face by area
        best = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
        x1, y1, x2, y2 = best.bbox.astype(int)
        crop = bgr[y1:y2, x1:x2]
        return best, crop

    def get_embedding_from_faceobj(self, face):
        emb = getattr(face, "embedding", None)
        if emb is None:
            return None
        emb = np.array(emb, dtype=np.float32)
        norm = np.linalg.norm(emb) + 1e-9
        return emb / norm

    def get_embedding_from_crop(self, crop):
        faces = self.get_faces(crop)
        if not faces:
            return None
        return self.get_embedding_from_faceobj(faces[0])

