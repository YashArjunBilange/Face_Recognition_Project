# main.py
import cv2
import numpy as np
from engine import FaceEngine
from database import FaceDatabase

THRESHOLD = 0.30   # cosine similarity threshold (0..1). Tune on your data.

def add_person_flow(engine, db):
    name = input("Enter person's name (no spaces): ").strip()
    if not name:
        print("Invalid name.")
        return

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Cannot open camera")
        return

    print("You will capture 3 images: FRONT, LEFT, RIGHT.")
    print("Position face accordingly and press 'c' to capture each one.")
    crops = []
    angle_names = ["front", "left", "right"]
    idx = 0

    while idx < 3:
        ret, frame = cam.read()
        if not ret:
            print("Frame read failed")
            break

        # show guide text
        cv2.putText(frame, f"Capture: {angle_names[idx]} - press c", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.imshow("Capture - press c", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            face_obj, crop = engine.best_face(frame)
            if face_obj is None:
                print("No face detected - try again")
                continue
            crops.append(face_obj.embedding)
            idx += 1
            print(f"Captured {angle_names[idx-1]}")
        elif key == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

    if len(crops) == 0:
        print("No images captured.")
        return

    try:
        db.add_person(name, crops)
        print("Person added.")
    except Exception as e:
        print("Failed to add person:", e)

def recognize_flow(engine, db):
    embeddings = db.load_all_embeddings()
    if not embeddings:
        print("No people in database. Add persons first.")
        return

    cam = cv2.VideoCapture(0)
    print("Recognition started. Press 'q' to quit.")
    while True:
        ret, frame = cam.read()
        if not ret:
            break

        face_obj, crop = engine.best_face(frame)
        if crop is not None and face_obj is not None:
            emb = face_obj.embedding.astype(np.float32)
            emb = emb / np.linalg.norm(emb)

            if emb is not None:
                # compute cosine with each saved name
                best_name = "Unknown"
                best_score = -1.0
                for name, saved in embeddings.items():
                    score = float(np.dot(emb, saved))
                    if score > best_score:
                        best_score = score
                        best_name = name
                # if best_score lower than threshold, mark Unknown
                if best_score < THRESHOLD:
                    best_name = "Unknown"

                x1, y1, x2, y2 = face_obj.bbox.astype(int)
                cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0),2)
                cv2.putText(frame, f"{best_name} {best_score:.2f}", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.imshow("Recognition", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

def manage_flow(db):
    while True:
        people = db.list_people()
        print("\nPeople in dataset:")
        for i, p in enumerate(people, start=1):
            print(f"{i}. {p}")
        print("a) Delete by number")
        print("b) Back")
        ch = input("choice: ").strip().lower()
        if ch == 'a':
            sel = input("Enter number to delete: ").strip()
            if sel.isdigit():
                idx = int(sel)-1
                if 0 <= idx < len(people):
                    name = people[idx]
                    db.delete_person(name)
                    print("Deleted", name)
                else:
                    print("Invalid number")
            else:
                print("Not a number")
        elif ch == 'b':
            break
        else:
            print("Unknown option")

def main():
    # If you have GPU and onnxruntime-gpu installed, set ctx_id=0
    engine = FaceEngine(ctx_id=-1, model_name="buffalo_l")
    db = FaceDatabase(engine)

    while True:
        print("\n=== Menu ===")
        print("1) Add Person (capture 3 images)")
        print("2) Recognize (webcam)")
        print("3) Manage dataset (list/delete)")
        print("4) Exit")
        c = input("Choose: ").strip()
        if c == "1":
            add_person_flow(engine, db)
        elif c == "2":
            recognize_flow(engine, db)
        elif c == "3":
            manage_flow(db)
        elif c == "4":
            break
        else:
            print("Invalid choice")

if __name__ == "__main__":
    main()
