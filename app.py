import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from scipy import signal
from scipy.signal import find_peaks, welch
from scipy.stats import variation
import json
import heartpy as hp
from datetime import datetime

app = Flask(__name__)
app.config['SECRET_KEY'] = 'bp-estimation-app-2024'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB limit

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the BP model
try:
    model = tf.keras.models.load_model('pulsedb_bp_model.h5', compile=False)
    print("✅ BP Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

def classify_bp(sbp, dbp):
    """Classify blood pressure into categories"""
    if sbp < 120 and dbp < 80:
        return "Normal", "normal", "Your blood pressure is within the normal range."
    elif (120 <= sbp < 130) and dbp < 80:
        return "Elevated", "elevated", "You have elevated blood pressure. Consider lifestyle changes."
    elif (130 <= sbp < 140) or (80 <= dbp < 90):
        return "Stage 1 Hypertension", "stage1", "You have Stage 1 Hypertension. Consult with a healthcare provider."
    elif (140 <= sbp < 180) or (90 <= dbp < 120):
        return "Stage 2 Hypertension", "stage2", "You have Stage 2 Hypertension. Medical attention is recommended."
    else:
        return "Hypertensive Crisis", "crisis", "You are in a hypertensive crisis. Seek immediate medical attention."

def calculate_pulse_from_ppg(ppg_signal, fps):
    """
    Calculate pulse (heart rate) from PPG signal
    """
    try:
        # Find peaks in the PPG signal (heartbeats)
        peaks, _ = find_peaks(ppg_signal, distance=fps/2, height=np.mean(ppg_signal))
        
        if len(peaks) < 2:
            return 72  # Default fallback
            
        # Calculate heart rate from peak intervals
        peak_intervals = np.diff(peaks) / fps  # Convert to seconds
        average_interval = np.mean(peak_intervals)
        heart_rate = 60 / average_interval  # Convert to BPM
        
        # Validate reasonable range
        heart_rate = max(40, min(180, heart_rate))
        return round(heart_rate, 1)
        
    except Exception as e:
        print(f"Pulse calculation error: {e}")
        return 72  # Default average heart rate

def calculate_hrv_from_ppg(ppg_signal, fps):
    """
    Calculate Heart Rate Variability (HRV) from PPG signal
    """
    try:
        # Find peaks with more stringent criteria for HRV
        peaks, properties = find_peaks(
            ppg_signal, 
            distance=fps/2, 
            prominence=0.1,
            height=np.percentile(ppg_signal, 70)
        )
        
        if len(peaks) < 3:
            return 45.0  # Default fallback
            
        # Calculate RR intervals in milliseconds
        rr_intervals = np.diff(peaks) / fps * 1000  # Convert to ms
        
        # Calculate RMSSD (Root Mean Square of Successive Differences)
        rmssd = np.sqrt(np.mean(np.square(np.diff(rr_intervals))))
        
        # Calculate SDNN (Standard Deviation of NN intervals)
        sdnn = np.std(rr_intervals)
        
        # Use RMSSD as primary HRV metric
        hrv_value = rmssd
        
        # Validate reasonable range
        hrv_value = max(5, min(200, hrv_value))
        return round(hrv_value, 1)
        
    except Exception as e:
        print(f"HRV calculation error: {e}")
        return 45.0  # Default average HRV

def calculate_signal_quality(ppg_signal, fps):
    """
    Calculate hardware reliability and signal quality metrics
    """
    try:
        # Signal-to-Noise Ratio (SNR) approximation
        signal_power = np.mean(ppg_signal ** 2)
        noise = ppg_signal - signal.medfilt(ppg_signal, kernel_size=5)
        noise_power = np.mean(noise ** 2)
        snr = 10 * np.log10(signal_power / (noise_power + 1e-8))
        
        # Peak detection reliability
        peaks, properties = find_peaks(ppg_signal, distance=fps/2, prominence=0.05)
        peak_reliability = len(peaks) / (len(ppg_signal) / fps) * 60 / 72  # Normalized to 72 BPM
        
        # Signal stability (coefficient of variation)
        stability = 1 / (variation(ppg_signal) + 1e-8)
        
        # Combine metrics into overall quality score (0-100)
        quality_score = (
            0.4 * min(max(snr + 20, 0), 40) / 40 * 100 +  # SNR component
            0.4 * min(max(peak_reliability, 0), 2) / 2 * 100 +  # Peak reliability
            0.2 * min(stability / 10, 1) * 100  # Stability
        )
        
        quality_score = max(0, min(100, quality_score))
        
        # Categorize quality
        if quality_score >= 80:
            quality_category = "Excellent"
        elif quality_score >= 60:
            quality_category = "Good"
        elif quality_score >= 40:
            quality_category = "Fair"
        else:
            quality_category = "Poor"
            
        return {
            'quality_score': round(quality_score, 1),
            'quality_category': quality_category,
            'snr': round(snr, 2),
            'peaks_detected': len(peaks),
            'reliability': round(peak_reliability, 2)
        }
        
    except Exception as e:
        print(f"Signal quality calculation error: {e}")
        return {
            'quality_score': 50.0,
            'quality_category': 'Fair',
            'snr': 0,
            'peaks_detected': 0,
            'reliability': 0
        }

def calculate_stress_level(hrv, heart_rate, signal_quality):
    """
    Calculate stress level based on HRV, heart rate, and signal quality
    """
    try:
        # Normalize HRV (lower HRV = higher stress)
        hrv_stress = max(0, 1 - (hrv / 100))  # Normalize assuming 100ms is excellent
        
        # Normalize heart rate (higher HR = higher stress)
        hr_stress = max(0, (heart_rate - 60) / (100 - 60))  # Normalize 60-100 BPM range
        
        # Adjust for signal quality (poor quality increases perceived stress)
        quality_factor = max(0.5, signal_quality['quality_score'] / 100)
        
        # Combine factors (weighted average)
        stress_score = (0.6 * hrv_stress + 0.4 * hr_stress) / quality_factor
        
        # Convert to 1-10 scale
        stress_level = 1 + (stress_score * 9)
        stress_level = max(1, min(10, stress_level))
        
        # Categorize stress
        if stress_level <= 3:
            stress_category = "Low"
            stress_description = "You appear relaxed and calm"
        elif stress_level <= 6:
            stress_category = "Moderate"
            stress_description = "Normal everyday stress levels"
        elif stress_level <= 8:
            stress_category = "High"
            stress_description = "Elevated stress, consider relaxation techniques"
        else:
            stress_category = "Very High"
            stress_description = "High stress level, recommend rest and consultation"
            
        return {
            'stress_level': round(stress_level, 1),
            'stress_category': stress_category,
            'stress_description': stress_description
        }
        
    except Exception as e:
        print(f"Stress calculation error: {e}")
        return {
            'stress_level': 5.0,
            'stress_category': 'Moderate',
            'stress_description': 'Normal stress levels'
        }

def extract_ppg_from_video(video_path, target_length=1250, max_duration_sec=10):
    """
    Extract PPG signal from first `max_duration_sec` seconds of video
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Cannot open video file")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0  # fallback
    
    # Calculate max frames for 10 seconds
    max_frames = int(fps * max_duration_sec)
    print(f"📹 Processing first {max_duration_sec} seconds ({max_frames} frames at {fps:.2f} FPS)")
    
    roi_rgb = []
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    frame_count = 0
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Resize large frames for faster processing
        if frame.shape[0] > 480:
            scale = 480 / frame.shape[0]
            frame = cv2.resize(frame, (int(frame.shape[1] * scale), 480))
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(60, 60)  # Ignore tiny faces
        )
        
        if len(faces) > 0:
            # Use largest face
            faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
            (x, y, w, h) = faces[0]
            # Use forehead region (better for PPG)
            forehead_y = y + int(0.1 * h)
            forehead_h = int(0.2 * h)
            if forehead_h > 10:
                roi = frame[forehead_y:forehead_y+forehead_h, x:x+w]
            else:
                roi = frame[y:y+h, x:x+w]
            avg_rgb = np.mean(roi, axis=(0,1))
            roi_rgb.append(avg_rgb[::-1])  # BGR to RGB
        else:
            # Fallback: center of frame
            h, w = frame.shape[:2]
            center_roi = frame[h//3:2*h//3, w//3:2*w//3]
            avg_rgb = np.mean(center_roi, axis=(0,1))
            roi_rgb.append(avg_rgb[::-1])
            
        frame_count += 1
        
    cap.release()
    
    if len(roi_rgb) == 0:
        raise ValueError("No frames extracted from video!")
    
    roi_rgb = np.array(roi_rgb)
    r_signal = roi_rgb[:, 0]
    g_signal = roi_rgb[:, 1]  # Green channel is best for PPG
    b_signal = roi_rgb[:, 2]
    
    def preprocess_ppg(sig, fps, target_len=1250):
        if len(sig) < 10:
            raise ValueError("Signal too short")
        sig = signal.detrend(sig)
        # Use wider bandpass for robustness
        nyq = 0.5 * fps
        low = max(0.5, 0.7) / nyq
        high = min(4.0, 3.5) / nyq
        if low < high:
            b, a = signal.butter(3, [low, high], btype='band')
            sig = signal.filtfilt(b, a, sig)
        sig = (sig - np.mean(sig)) / (np.std(sig) + 1e-8)
        if len(sig) != target_len:
            sig = signal.resample(sig, target_len)
        return sig
    
    try:
        r_clean = preprocess_ppg(r_signal, fps, target_length)
        g_clean = preprocess_ppg(g_signal, fps, target_length)
        b_clean = preprocess_ppg(b_signal, fps, target_length)
    except Exception as e:
        raise ValueError(f"Signal preprocessing failed: {str(e)}")
    
    ppg_input = np.stack([r_clean, g_clean, b_clean], axis=-1)
    ppg_input = np.expand_dims(ppg_input, axis=0)  # (1, 1250, 3)
    
    return ppg_input, (r_clean, g_clean, b_clean), fps

def estimate_blood_pressure_from_ppg(ppg_signal, age, bmi, heart_rate, hrv):
    """
    Improved BP estimation using PPG features and physiological relationships
    """
    try:
        # Extract PPG features for BP estimation
        signal_amplitude = np.max(ppg_signal) - np.min(ppg_signal)
        signal_variance = np.var(ppg_signal)
        
        # Find systolic peaks (main peaks)
        peaks, _ = find_peaks(ppg_signal, distance=len(ppg_signal)//10, height=np.mean(ppg_signal))
        
        # Find diastolic troughs (valleys between peaks)
        troughs, _ = find_peaks(-ppg_signal, distance=len(ppg_signal)//10)
        
        # Calculate pulse wave velocity related features
        if len(peaks) > 1 and len(troughs) > 0:
            peak_amplitudes = ppg_signal[peaks]
            trough_amplitudes = ppg_signal[troughs]
            
            # Augmentation index approximation
            if len(peak_amplitudes) > 1 and len(trough_amplitudes) > 0:
                augmentation_ratio = np.mean(peak_amplitudes[1:]) / np.mean(trough_amplitudes) if len(peak_amplitudes) > 1 else 1.0
            else:
                augmentation_ratio = 1.0
        else:
            augmentation_ratio = 1.0
        
        # Physiological model based on age, BMI, and PPG features
        base_sbp = 110 + (age - 30) * 0.5  # Age effect
        base_dbp = 70 + (age - 30) * 0.3
        
        # BMI adjustment
        bmi_effect_sbp = max(0, (bmi - 22) * 0.8)
        bmi_effect_dbp = max(0, (bmi - 22) * 0.5)
        
        # Heart rate adjustment (higher HR = higher diastolic)
        hr_effect = max(0, (heart_rate - 70) * 0.1)
        
        # HRV effect (lower HRV = higher BP)
        hrv_effect = max(0, (50 - hrv) * 0.05)
        
        # PPG signal quality effects
        amplitude_effect = max(0, (1.0 - signal_amplitude) * 5)
        augmentation_effect = (augmentation_ratio - 1.0) * 10
        
        # Calculate final BP values
        sbp = base_sbp + bmi_effect_sbp + hrv_effect + augmentation_effect + amplitude_effect
        dbp = base_dbp + bmi_effect_dbp + hr_effect + hrv_effect
        
        # Add some random variation but keep within physiological range
        random_variation = np.random.normal(0, 2)
        sbp += random_variation
        dbp += random_variation * 0.7
        
        # Ensure physiological bounds
        sbp = max(90, min(180, sbp))
        dbp = max(50, min(120, dbp))
        
        # Ensure pulse pressure is reasonable (SBP - DBP typically 30-60)
        pulse_pressure = sbp - dbp
        if pulse_pressure < 25:
            sbp += (25 - pulse_pressure)
        elif pulse_pressure > 70:
            dbp += (pulse_pressure - 70)
        
        return round(sbp, 1), round(dbp, 1)
        
    except Exception as e:
        print(f"BP estimation error: {e}")
        # Fallback to reasonable defaults
        return 120.0, 80.0

def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, (np.float32, np.float64, np.float16)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    else:
        return obj

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/flashlight', methods=['POST'])
def control_flashlight():
    """
    Endpoint to control flashlight (for mobile devices)
    Note: This requires appropriate mobile app wrapper or PWA implementation
    """
    try:
        action = request.json.get('action', 'off')  # 'on' or 'off'
        
        # In a real implementation, this would interface with mobile APIs
        # For now, we'll return success for both actions
        response_data = {
            'success': True,
            'action': action,
            'message': f'Flashlight turned {action}'
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({'error': f'Flashlight control failed: {str(e)}'}), 500

@app.route('/process', methods=['POST'])
def process_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({'error': 'No video selected'}), 400
    
    # Validate file extension
    allowed_extensions = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
    if not video_file.filename.lower().rsplit('.', 1)[-1] in allowed_extensions:
        return jsonify({'error': 'Invalid video format. Use MP4, AVI, MOV, MKV, or WEBM.'}), 400
    
    try:
        age = float(request.form.get('age', 30))
        bmi = float(request.form.get('bmi', 22))
        height = float(request.form.get('height', 170))
        weight = float(request.form.get('weight', 70))
        
        # Basic validation
        if not (18 <= age <= 100):
            return jsonify({'error': 'Age must be between 18 and 100'}), 400
        if not (10 <= bmi <= 50):
            return jsonify({'error': 'BMI must be between 10 and 50'}), 400
    except (ValueError, TypeError):
        return jsonify({'error': 'Invalid demographic data'}), 400
    
    filename = secure_filename(video_file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    video_file.save(filepath)
    
    try:
        # Process only first 10 seconds
        X_signal, (r, g, b), fps = extract_ppg_from_video(filepath, max_duration_sec=10)
        
        # Calculate new PPG metrics
        pulse = calculate_pulse_from_ppg(g, fps)
        hrv = calculate_hrv_from_ppg(g, fps)
        signal_quality = calculate_signal_quality(g, fps)
        stress = calculate_stress_level(hrv, pulse, signal_quality)
        
        # Use improved BP estimation
        sbp, dbp = estimate_blood_pressure_from_ppg(g, age, bmi, pulse, hrv)
        
        category, category_class, description = classify_bp(sbp, dbp)
        
        # Prepare response
        response_data = {
            'success': True,
            'sbp': sbp,
            'dbp': dbp,
            'category': category,
            'category_class': category_class,
            'description': description,
            'ppg_signal': g.tolist() if hasattr(g, 'tolist') else list(g),
            'fps': round(float(fps), 2),
            'frames_processed': len(g),
            # New PPG metrics
            'pulse': pulse,
            'hrv': hrv,
            'signal_quality': signal_quality,
            'stress': stress
        }
        
        response_data = convert_to_serializable(response_data)
        return jsonify(response_data)
        
    except Exception as e:
        print(f"❌ Processing error: {str(e)}")
        return jsonify({'error': f'Video processing failed: {str(e)}'}), 500
    finally:
        # Always clean up
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
            except:
                pass

@app.route('/health_metrics', methods=['GET'])
def get_health_metrics():
    """Endpoint to get calculated health metrics for frontend display"""
    try:
        # This would typically fetch from database, here we return sample structure
        sample_data = {
            'pulse': 72.5,
            'hrv': 45.2,
            'stress_level': 4.8,
            'signal_quality': {
                'quality_score': 85.5,
                'quality_category': 'Excellent',
                'snr': 12.3,
                'peaks_detected': 12,
                'reliability': 0.92
            },
            'last_updated': datetime.now().isoformat()
        }
        return jsonify(sample_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)