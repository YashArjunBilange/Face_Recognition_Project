# database.py
import os
import numpy as np
from pathlib import Path
import shutil

DATASET_DIR = Path("dataset")

class FaceDatabase:
    def __init__(self, engine):
        """
        engine: FaceEngine instance (for computing embeddings on captured images)
        """
        self.engine = engine
        DATASET_DIR.mkdir(exist_ok=True)

    def add_person(self, name, embeddings):
        person_dir = DATASET_DIR / name
        person_dir.mkdir(parents=True, exist_ok=True)

        emb_list = []
        for i, emb in enumerate(embeddings, start=1):
            emb = np.asarray(emb, dtype=np.float32)
            n = np.linalg.norm(emb)
            if n > 0:
                emb = emb / n
            emb_list.append(emb)

        if len(emb_list) == 0:
            shutil.rmtree(person_dir)
            raise ValueError("No valid embeddings")

        avg = np.mean(np.vstack(emb_list), axis=0)
        avg = avg / np.linalg.norm(avg)
        np.save(person_dir / "embedding.npy", avg)
        print(f"[db] saved {name} with {len(emb_list)} embeddings")

    def load_all_embeddings(self):
        """
        Returns dict name -> embedding (normalized)
        """
        data = {}
        for p in DATASET_DIR.iterdir():
            if p.is_dir():
                emb_file = p / "embedding.npy"
                if emb_file.exists():
                    emb = np.load(emb_file)
                    # ensure normalized
                    n = np.linalg.norm(emb)
                    if n>0:
                        emb = emb / n
                    data[p.name] = emb
        return data

    def list_people(self):
        return sorted([p.name for p in DATASET_DIR.iterdir() if p.is_dir()])

    def delete_person(self, name):
        p = DATASET_DIR / name
        if p.exists() and p.is_dir():
            shutil.rmtree(p)
            return True
        return False
