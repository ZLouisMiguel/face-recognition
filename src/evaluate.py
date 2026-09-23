import numpy as np


def cosine_distance(emb1, emb2):
    return 1.0 - np.dot(emb1, emb2)


def evaluate_thresholds(db_path="data/db/face_db.npz"):
    db = np.load(db_path)
    keys = list(db.keys())
    print(f"Evaluated {len(keys)} registered identities.")
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            dist = cosine_distance(db[keys[i]], db[keys[j]])
            print(f"Distance between {keys[i]} and {keys[j]}: {dist:.4f}")


if __name__ == "__main__":
    evaluate_thresholds()
