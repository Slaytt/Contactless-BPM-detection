import numpy as np

class SignalProcessor:
    def __init__(self, buffer_size=300):
        self.buffer_size = buffer_size
        self.signal_buffer = []
        self.bpm_history = []

    def process_value(self, value, current_fps):
        self.signal_buffer.append(value)
        
        if len(self.signal_buffer) > self.buffer_size:
            self.signal_buffer.pop(0)
            
        if len(self.signal_buffer) == self.buffer_size:
            return self._compute_bpm(current_fps)
        
        return None

    def _compute_bpm(self, fps):
        raw_signal = np.array(self.signal_buffer)
        valid_fps = fps if fps > 10 else 30.0

        # Detrending
        x = np.arange(len(raw_signal))
        p = np.polyfit(x, raw_signal, 1)
        trend = np.polyval(p, x)
        detrended = raw_signal - trend
        
        # Normalisation
        normalized = (detrended - np.mean(detrended)) / (np.std(detrended) + 1e-5)
        
        # FFT
        fft_spectrum = np.fft.rfft(normalized)
        fft_freqs = np.fft.rfftfreq(len(normalized), d=1.0/valid_fps)
        fft_mags = np.abs(fft_spectrum)
        
        # Filtrage
        min_bpm, max_bpm = 45.0, 200.0
        mask = (fft_freqs >= min_bpm/60.0) & (fft_freqs <= max_bpm/60.0)
        
        valid_freqs = fft_freqs[mask]
        valid_mags = fft_mags[mask]
        
        if len(valid_mags) > 0:
            peak_idx = np.argmax(valid_mags)
            bpm_instant = valid_freqs[peak_idx] * 60.0
            return self._smooth_bpm(bpm_instant)
        
        return None

    def _smooth_bpm(self, new_bpm):
        self.bpm_history.append(new_bpm)
        if len(self.bpm_history) > 30:
            self.bpm_history.pop(0)
        return np.mean(self.bpm_history)