import os
import pickle
import mediapipe as mp
import cv2
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)

# Set dataset directory
DATA_DIR = 'Project_part_4/data'

data = []
labels = []

# Iterate over each label folder
for label in os.listdir(DATA_DIR):
    label_path = os.path.join(DATA_DIR, label)
    if not os.path.isdir(label_path):
        continue

    for img_name in os.listdir(label_path):
        img_path = os.path.join(label_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            data_aux = []
            x_, y_ = [], []

            for hand_landmarks in results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    x_.append(landmark.x)
                    y_.append(landmark.y)
                
                for landmark in hand_landmarks.landmark:
                    data_aux.append(landmark.x - min(x_))
                    data_aux.append(landmark.y - min(y_))
            
            data.append(data_aux)
            labels.append(label)

# Padding data to ensure consistency
max_length = max(len(sample) for sample in data)
data_padded = np.array([sample + [0] * (max_length - len(sample)) for sample in data])

# Save dataset as a pickle file
with open('Project_part_4/data.pickle', 'wb') as f:
    pickle.dump({'data': data_padded, 'labels': labels}, f)

print("✅ Dataset created successfully!")
