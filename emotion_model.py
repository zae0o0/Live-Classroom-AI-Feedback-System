import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN

try:
    from fer import FER
except (ModuleNotFoundError, ImportError):
    import sys, types
    sys.modules["moviepy"] = types.ModuleType("moviepy")
    sys.modules["moviepy.editor"] = types.ModuleType("moviepy.editor")
    from fer import FER

emotion_detector = FER(mtcnn=True)
face_memory = {}
next_id = 1

def map_to_classroom_emotion(fer_emotion):
    if fer_emotion in ["happy", "surprise"]:
        return "Engaged"
    elif fer_emotion == "neutral":
        return "Confused"
    elif fer_emotion in ["sad", "angry", "fear", "disgust"]:
        return "Bored/Frustrated"
    else:
        return "Unknown"

def get_face_embedding(face_crop):
    resized_face = cv2.resize(face_crop, (48, 48))
    gray = cv2.cvtColor(resized_face, cv2.COLOR_RGB2GRAY)
    return gray.flatten()

def assign_id(face_embedding):
    global next_id
    similarity_threshold = 10000
    for fid, existing_embedding in face_memory.items():
        dist = np.linalg.norm(face_embedding - existing_embedding)
        if dist < similarity_threshold:
            return fid
    new_id = next_id
    face_memory[new_id] = face_embedding
    next_id += 1
    return new_id

def detect_emotion(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detected_faces = emotion_detector.detect_emotions(rgb_frame)
    results = []

    for face in detected_faces:
        x, y, w, h = face["box"]
        x, y = abs(x), abs(y)
        face_crop = rgb_frame[y:y+h, x:x+w]
        if face_crop.size == 0:
            continue
        embedding = get_face_embedding(face_crop)
        face_id = assign_id(embedding)
        emotions = face["emotions"]
        if emotions:
            top_fer_emotion = max(emotions, key=emotions.get)
            classroom_emotion = map_to_classroom_emotion(top_fer_emotion)
            score = emotions[top_fer_emotion]
        else:
            classroom_emotion = "Unknown"
            score = 0.0
        results.append({
            "id": int(face_id),
            "box": [int(x), int(y), int(w), int(h)],
            "emotion": classroom_emotion,
            "score": float(score)
        })
    return results
