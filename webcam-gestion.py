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

#keep track of the last time the signal was updated
last_time = time.time()

signal_buffer = []

#to store the last 30 bpm values
bpm_history = []
# Use context manager to handle resource cleanup
with FaceLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # calculate fps
        current_time = time.time()
        delta_time = current_time - last_time
        last_time = current_time

        if delta_time > 0: 
            fps = 1 / delta_time
        print(f"FPS: {fps}")
        
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
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                mean_green = np.mean(roi[:, :, 1])
                signal_buffer.append(mean_green)
                
                BUFFER_SIZE = 300
                if len(signal_buffer) > BUFFER_SIZE:
                    signal_buffer.pop(0)
                
                if len(signal_buffer) == BUFFER_SIZE:
                    
                    raw_signal = np.array(signal_buffer)
                    
                    avg_fps = 30.0 
                    if fps > 10: 
                        avg_fps = fps

                    x = np.arange(len(raw_signal))
                    p = np.polyfit(x, raw_signal, 1) 
                    trend = np.polyval(p, x)         
                    detrended_signal = raw_signal - trend 
                    
                    normalized_signal = (detrended_signal - np.mean(detrended_signal)) / (np.std(detrended_signal) + 1e-5)
                    
                    fft_spectrum = np.fft.rfft(normalized_signal)
                    fft_freqs = np.fft.rfftfreq(len(normalized_signal), d=1.0/avg_fps)
                    fft_magnitude = np.abs(fft_spectrum)

                    min_bpm, max_bpm = 45.0, 200.0
                    mask = (fft_freqs >= min_bpm/60.0) & (fft_freqs <= max_bpm/60.0)
                    
                    valid_freqs = fft_freqs[mask]
                    valid_mags = fft_magnitude[mask]

                    if len(valid_mags) > 0:
                        peak_index = np.argmax(valid_mags)
                        dominant_freq = valid_freqs[peak_index]
                        bpm_instantane = dominant_freq * 60.0
                        
                        bpm_history.append(bpm_instantane)
                        
                        if len(bpm_history) > 30:
                            bpm_history.pop(0)
                            
                        bpm_smooth = np.mean(bpm_history)
                        
                        text_bpm = f"BPM: {int(bpm_smooth)}"
                        
                        color = (0, 255, 0) if 50 < bpm_smooth < 100 else (0, 0, 255)
                        
                        cv2.putText(frame, text_bpm, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                        
                        print(f"Instantané: {bpm_instantane:.1f} | Lissé: {bpm_smooth:.1f}")
        cv2.imshow("Frame", frame)
        pressedKey = cv2.waitKey(1)
        if pressedKey == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
