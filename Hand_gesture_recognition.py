import cv2 
import mediapipe as mp
import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

DATA_PATH = "hand_sign_data.csv"
MODEL_PATH = "hand_sign_model.pkl"

# Change/expand this dict for your own gestures: number key/label -> named gesture string
GESTURE_LABELS = {
    0: "Fist",
    1: "Palm Open",
    2: "Thumbs Up",
    3: "Peace",
    4: "Okay",
    5: "Rock",
    6: "Spiderman",
    7: "L Shape",
    8: "Four",
    9: "Three"
}


# Reverse lookup for display
GESTURE_NAMES = {v: k for k, v in GESTURE_LABELS.items()}

def collect_data():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1)
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    collected = []
    print("\n=== DATA COLLECTION MODE ===")
    print("Show a gesture and press its number key (0-9) to label & save sample me the files.")
    print("Labels:")
    for idx, name in GESTURE_LABELS.items():
        print(f"{idx}: {name}")
    print("Press 'q' to quit and save all data.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        landmarks = [0] * (21*2)
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            for idx, lm in enumerate(hand_landmarks.landmark):
                landmarks[2*idx] = lm.x
                landmarks[2*idx+1] = lm.y
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("Collect Data - Hand Gestures", frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif 48 <= key <= 57 and any(landmarks):  # keys '0'-'9'
            idx = key - 48
            if idx in GESTURE_LABELS:
                collected.append(landmarks + [idx])
                print(f"Captured: {GESTURE_LABELS[idx]} [{idx}] (total {len(collected)})")

    cap.release()
    cv2.destroyAllWindows()
    if collected:
        df = pd.DataFrame(collected)
        header = not os.path.exists(DATA_PATH)
        df.to_csv(DATA_PATH, mode='a', header=header, index=False)
        print(f"\nSaved {len(collected)} samples to {DATA_PATH}")
    else:
        print("No samples collected.")

def train_model():
    print("\n=== TRAINING MODE ===")
    if not os.path.exists(DATA_PATH):
        print("No dataset found. Please collect data first.")
        return

    data = pd.read_csv(DATA_PATH)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=250)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(clf, f)
    print(f"Model trained! Test accuracy: {score*100:.2f}%")
    print(f"Model saved to {MODEL_PATH}")

def recognize():
    print("\n=== REAL-TIME RECOGNITION MODE ===")
    if not os.path.exists(MODEL_PATH):
        print("No model found. Please train the model first.")
        return

    with open(MODEL_PATH, "rb") as f:
        clf = pickle.load(f)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1)
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        landmarks = [0] * (21*2)
        predicted_label = None

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            for idx, lm in enumerate(hand_landmarks.landmark):
                landmarks[2*idx] = lm.x
                landmarks[2*idx+1] = lm.y
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            if any(landmarks):
                label = int(clf.predict([landmarks])[0])
                pred = GESTURE_LABELS.get(label, str(label))
                cv2.putText(frame, f'Predicted: {pred}',
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)

        cv2.imshow("Hand Gesture Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    print("Hand Gesture Recognition - Unified Script")
    print("Select Mode:")
    print("1) Collect gesture data")
    print("2) Train classifier")
    print("3) Run real-time recognition")
    mode = input("Enter [1/2/3]: ").strip()
    if mode == '1':
        collect_data()
    elif mode == '2':
        train_model()
    elif mode == '3':
        recognize()
    else:
        print("Invalid option.")

if __name__ == "__main__":
    main()
