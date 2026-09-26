import cv2
import os
import pickle
from deepface import DeepFace
import numpy as np

DATASET_PATH = "dataset"   # folder: dataset/Aarush, dataset/Avishka

db = {}

print("Starting encoding...")

for person_name in os.listdir(DATASET_PATH):
    person_path = os.path.join(DATASET_PATH, person_name)

    if not os.path.isdir(person_path):
        continue

    embeddings = []

    print("Processing:", person_name)

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)

        try:
            embedding = DeepFace.represent(
                img_path=img_path,
                model_name="Facenet",
                enforce_detection=True
            )[0]["embedding"]

            embeddings.append(embedding)

        except Exception as e:
            print("Skipping:", img_path)

    # FIX: store MEAN embedding (VERY IMPORTANT)
    if len(embeddings) > 0:
        db[person_name] = np.mean(np.array(embeddings), axis=0)
    else:
        print(f"No embeddings found for {person_name}")

# save
with open("face_db.pkl", "wb") as f:
    pickle.dump(db, f)

print("DONE → face_db.pkl created")