# Dự án nhận diện cử chỉ tay điều khiển trình chiếu

## Mục tiêu
Hệ thống nhận diện 21 điểm bàn tay theo thời gian thực, chuyển đổi sang chuỗi token (K-Means), rồi dùng LSTM để dự đoán cử chỉ. Khi nhận diện đúng, chương trình điều khiển PowerPoint bằng phím tắt.

## Thành phần chính
- [app.py](app.py): Chạy nhận diện realtime, điều khiển trình chiếu.
- [Pipeline.ipynb](Pipeline.ipynb): Notebook train K-Means + LSTM, xử lý dữ liệu JESTER.
- [hand_landmarker.task](hand_landmarker.task): Model MediaPipe Hand Landmarker.
- [hand_pose_kmeans.pkl](hand_pose_kmeans.pkl): K-Means tạo “từ điển” tư thế tay.
- [gesture_lstm_t4.pth](gesture_lstm_t4.pth): Trọng số LSTM đã train.
- [requirements.txt](requirements.txt): Danh sách thư viện.

## Cài đặt
```powershell
pip install -r requirements.txt
```

Tải model MediaPipe (nếu chưa có):
```powershell
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task" -OutFile "hand_landmarker.task"
```

## Chạy demo realtime
```powershell
python app.py
```

### Thử trực tiếp với PowerPoint
1. Mở file `.pptx` và vào trình chiếu (F5).
2. Chạy `python app.py`.
3. Dùng cử chỉ để điều khiển:
	 - `Swiping Right` → Next slide
	 - `Swiping Left` → Previous slide
	 - `Stop Sign` → Pause/Stop

Lưu ý: `pyautogui` sẽ gửi phím thật, nên hãy đóng các cửa sổ không liên quan trước khi test.

## Tinh chỉnh chất lượng nhận diện
Trong [app.py](app.py) có thể chỉnh:
- `min_hand_detection_confidence`, `min_tracking_confidence`: tăng/giảm độ nhạy detection.
- `recent_tokens` (cửa sổ làm mượt token): tăng để ổn định hơn, giảm để nhanh hơn.
- Ngưỡng `confidence` của LSTM (mặc định 0.75).

## Quy trình train (tóm tắt)
Chi tiết nằm trong [Pipeline.ipynb](Pipeline.ipynb):
1. Tải và lọc dataset 20BN-JESTER (Kaggle).
2. Trích xuất 21 landmarks từ frames bằng MediaPipe Tasks.
3. Train K-Means để tạo 32 cụm tư thế tay.
4. Mã hoá video thành chuỗi token 16 frame.
5. Train LSTM để phân loại cử chỉ.

## Ghi chú môi trường
- Cần webcam.
- Nếu gặp cảnh báo oneDNN (TensorFlow), có thể tắt bằng:
	```powershell
	$env:TF_ENABLE_ONEDNN_OPTS=0
	```

## Tác giả
Dinox16
## Github
