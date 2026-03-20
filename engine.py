import face_recognition
import numpy as np

class FaceEngine:

    def get_faces(self, rgb_img):
        boxes = face_recognition.face_locations(rgb_img)
        encodings = face_recognition.face_encodings(rgb_img, boxes)
        return boxes, encodings

    def best_face(self, rgb_img):
        boxes, encodings = self.get_faces(rgb_img)
        if not boxes:
            return None, None
        return boxes[0], encodings[0]

    def get_embedding_from_faceobj(self, encoding):
        return np.array(encoding, dtype=np.float32)
