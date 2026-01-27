import cv2
import mediapipe as mp
import numpy as np
import time

# Import MediaPipe Task API
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

current_result = None

# callback function to print the result
def print_result(result: FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global current_result
    current_result = result

# Initialize FaceLandmarker
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='face_landmarker.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_faces=1,
    result_callback=print_result)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

signal_buffer = []
# Use context manager to handle resource cleanup
with FaceLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_timestamp_ms = int(time.time() * 1000)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        landmarker.detect_async(mp_image, frame_timestamp_ms)
            
        if current_result and current_result.face_landmarks:
            landmarks = current_result.face_landmarks[0]
            
            h, w, _ = frame.shape
            
            cx = int(landmarks[10].x * w)
            cy = int(landmarks[10].y * h)
            
            box_size = 40
            half = box_size // 2
            
            x1 = max(0, cx - half)
            y1 = max(0, cy - half)
            x2 = min(w, cx + half)
            y2 = min(h, cy + half)
            
            roi = frame[y1:y2, x1:x2]

            if roi.size > 0:
                    # Draw the square 
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    mean_green = np.mean(roi[:, :, 1])
                    
                    # store in the list
                    signal_buffer.append(mean_green)
                    
                    # limit the size of the list
                    if len(signal_buffer) > 300:
                        signal_buffer.pop(0)

                    print(f"Signal Vert: {mean_green:.2f} | Buffer: {len(signal_buffer)}")

        cv2.imshow("Frame", frame)
        pressedKey = cv2.waitKey(1)
        if pressedKey == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
