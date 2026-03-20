import numpy as np
from pathlib import Path

class FaceDatabase:
    def __init__(self, db_path="faces"):
        self.db_path = Path(db_path)
        self.db_path.mkdir(exist_ok=True)

    def person_file(self, name):
        return self.db_path / f"{name}.npy"

    def add_person(self, name, embeddings):
        emb = np.array(embeddings, dtype=np.float32)
        mean_emb = emb.mean(axis=0)
        mean_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-9)
        np.save(self.person_file(name), mean_emb)

    def list_people(self):
        return [p.stem for p in self.db_path.glob("*.npy")]

    def delete_person(self, name):
        f = self.person_file(name)
        if f.exists():
            f.unlink()

    def load_all_embeddings(self):
        data = {}
        for p in self.db_path.glob("*.npy"):
            emb = np.load(p)
            emb = emb / (np.linalg.norm(emb) + 1e-9)
            data[p.stem] = emb
        return data
