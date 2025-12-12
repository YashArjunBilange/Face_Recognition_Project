import os
import numpy as np
from pathlib import Path
from PIL import Image

class FaceDatabase:
    def __init__(self, engine, db_path="faces"):
        self.engine = engine
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)

    def person_file(self, name):
        return str(self.db_path / f"{name}.npy")

    def image_folder(self, name):
        p = self.db_path / name
        p.mkdir(exist_ok=True)
        return str(p)

    def add_person(self, name, embeddings, raw_images=None):
        """
        embeddings: list/array of embeddings (already normalized)
        raw_images: optional list of BGR images to save
        """
        emb = np.array(embeddings, dtype=np.float32)
        # store the mean embedding as the canonical embedding
        mean_emb = emb.mean(axis=0)
        mean_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-9)
        np.save(self.person_file(name), mean_emb)

        if raw_images:
            folder = Path(self.image_folder(name))
            for i, img in enumerate(raw_images):
                # img is expected to be BGR numpy array
                im = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                im.save(folder / f"{i+1}.jpg")

    def list_people(self):
        return sorted([p.stem for p in self.db_path.glob("*.npy")])

    def delete_person(self, name):
        f = Path(self.person_file(name))
        if f.exists():
            f.unlink()
        folder = self.db_path / name
        if folder.exists() and folder.is_dir():
            for item in folder.iterdir():
                item.unlink()
            folder.rmdir()

    def load_all_embeddings(self):
        data = {}
        for p in self.db_path.glob("*.npy"):
            emb = np.load(str(p))
            emb = emb / (np.linalg.norm(emb) + 1e-9)
            data[p.stem] = emb
        return data
