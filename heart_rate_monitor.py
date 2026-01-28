import cv2
import time
import numpy as np

# IMPORT DES AUTRES FICHIERS
from face_tracking import FaceTracker
from signal_processor import SignalProcessor

class HeartRateMonitor:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise ValueError("Erreur Webcam")
            
        self.detector = FaceTracker()
        self.processor = SignalProcessor(buffer_size=300)
        
        self.last_time = time.time()
        self.fps = 0.0

    def calculate_fps(self):
        current_time = time.time()
        delta = current_time - self.last_time
        self.last_time = current_time
        if delta > 0:
            self.fps = 1.0 / delta
        return self.fps

    def run(self):
        print("Démarrage du moniteur... (Appuie sur 'q' pour quitter)")
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret: break
                
                fps = self.calculate_fps()
                timestamp = int(time.time() * 1000)

                self.detector.detect_async(frame, timestamp)
                roi, coords = self.detector.get_forehead_roi(frame)
                
                if roi is not None and roi.size > 0:
                    x1, y1, x2, y2 = coords
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
\
                    mean_green = np.mean(roi[:, :, 1])
                    bpm = self.processor.process_value(mean_green, fps)
                    
                    if bpm is not None:
                        text = f"BPM: {int(bpm)}"
                        color = (0, 255, 0) if 50 < bpm < 100 else (0, 0, 255)
                        cv2.putText(frame, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                        print(f"FPS: {fps:.1f} | BPM: {bpm:.1f}")
                    else:
                        pct = int((len(self.processor.signal_buffer) / self.processor.buffer_size) * 100)
                        cv2.putText(frame, f"Calibration: {pct}%", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                cv2.imshow("rPPG Monitor Modular", frame)
                if cv2.waitKey(1) == ord('q'):
                    break
        finally:
            self.cap.release()
            self.detector.close()
            cv2.destroyAllWindows()