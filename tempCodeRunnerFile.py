import cv2
import mediapipe as mp
import torch
import numpy as np
import joblib
from collections import deque
import pyautogui
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
class GestureLSTM(nn.Module):
    def __init__(self, vocab_size=33, embed_dim=128, hidden_dim=256, num_classes=4):
        super(GestureLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, num_layers=2, dropout=0.3)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.embedding(x)
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1]) 
        return out

# 2. LOAD CÁC THÀNH PHẦN ĐÃ TRAIN
DEVICE = torch.device('cpu') # Chạy trên Acer dùng CPU cho ổn định
CLASSES = ['Stop Sign', 'No gesture', 'Swiping Right', 'Swiping Left']

kmeans = joblib.load('hand_pose_kmeans.pkl')
model = GestureLSTM(num_classes=len(CLASSES))
model.load_state_dict(torch.load('gesture_lstm_t4.pth', map_location=DEVICE))
model.eval()

# 3. CẤU HÌNH MEDIAPIPE & BUFFER
base_options = python.BaseOptions(
    model_asset_path='hand_landmarker.task',
    delegate=python.BaseOptions.Delegate.CPU
)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)
seq_buffer = deque(maxlen=16) # Buffer chứa 16 tokens liên tiếp
cooldown = 0 # Tránh nhận diện 1 hành động quá nhiều lần liên tục

cap = cv2.VideoCapture(0)

print("Hệ thống đã sẵn sàng! Đưa tay lên xem nào Khánh...")

while cap.isOpened():
    success, img = cap.read()
    if not success: break
    
    img = cv2.flip(img, 1) # Lật ảnh như gương
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    token = 32 # Mặc định là No Hand
    
    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)
            
            # Trích xuất & Chuẩn hóa tọa độ (Wrist làm gốc)
            coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms.landmark])
            rel_coords = coords - coords[0]
            max_val = np.abs(rel_coords).max()
            if max_val > 0: rel_coords /= max_val
            
            # K-means dự đoán ID tư thế
            token = kmeans.predict(rel_coords.flatten().reshape(1, -1))[0]
            
    seq_buffer.append(token)
    
    # 4. DỰ ĐOÁN HÀNH ĐỘNG KHI ĐỦ 16 FRAMES
    if len(seq_buffer) == 16 and cooldown == 0:
        input_seq = torch.LongTensor([list(seq_buffer)]).to(DEVICE)
        with torch.no_grad():
            output = model(input_seq)
            probs = torch.softmax(output, dim=1)
            confidence, prediction = torch.max(probs, dim=1)
            
        if confidence.item() > 0.8: # Chỉ thực hiện nếu tin cậy trên 80%
            gesture = CLASSES[prediction.item()]
            
            # --- LOGIC ĐIỀU KHIỂN THỰC TẾ ---
            if gesture == 'Swiping Right':
                print("NEXT SLIDE")
                pyautogui.press('right')
                cooldown = 15 # Đợi 15 frames tiếp theo mới nhận diện tiếp
            elif gesture == 'Swiping Left':
                print("PREVIOUS SLIDE")
                pyautogui.press('left')
                cooldown = 15
            elif gesture == 'Stop Sign':
                print("PAUSE/STOP")
                pyautogui.press('space') # Tạm dừng video/slide
                cooldown = 20
                
            cv2.putText(img, f"Gesture: {gesture}", (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if cooldown > 0: cooldown -= 1
    
    cv2.imshow("DTU Gesture Control - Khanh Ho", img)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()