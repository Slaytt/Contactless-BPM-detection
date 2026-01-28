import cv2
import mediapipe as mp

class FaceTracker:
    def __init__(self):
        # Configuration MediaPipe
        self.BaseOptions = mp.tasks.BaseOptions
        self.FaceLandmarker = mp.tasks.vision.FaceLandmarker
        self.FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        self.VisionRunningMode = mp.tasks.vision.RunningMode
        
        self.latest_result = None

        options = self.FaceLandmarkerOptions(
            base_options=self.BaseOptions(model_asset_path='face_landmarker.task'),
            running_mode=self.VisionRunningMode.LIVE_STREAM,
            num_faces=1,
            result_callback=self._callback_result)
        
        self.landmarker = self.FaceLandmarker.create_from_options(options)

    def _callback_result(self, result, output_image, timestamp_ms):
        self.latest_result = result

    def detect_async(self, frame, timestamp_ms):
        # Conversion BGR -> RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        self.landmarker.detect_async(mp_image, timestamp_ms)

    def get_forehead_roi(self, frame):
        if self.latest_result and self.latest_result.face_landmarks:
            landmarks = self.latest_result.face_landmarks[0]
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
            return roi, (x1, y1, x2, y2)
        
        return None, None

    def close(self):
        self.landmarker.close()