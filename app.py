import os
import platform
import torch
import numpy as np
from flask import Flask, request, jsonify, render_template
from nemo.collections.asr.models.msdd_models import NeuralDiarizer
from flask_cors import CORS
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Back‑port sctypes for backward compatibility
np.sctypes = {
    'int': [np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64],
    'uint': [np.uint8, np.uint16, np.uint32, np.uint64, np.uint],
    'float': [np.float16, np.float32, np.float64],
    'complex': [np.complex64, np.complex128]
}

# 1. Init Flask app
app = Flask(__name__)
UPLOAD_FOLDER = "./uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

CORS(app)  # Enable CORS for all routes
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

class SpeakerDiarizer:
    def __init__(self, audio_file: str, model_name: str = "diar_msdd_telephonic", vad_model_name: str = "vad_multilingual_marblenet"):
        self.audio_file = audio_file
        self.model_name = model_name
        self.vad_model_name = vad_model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.annotation = None
        self.rttm_file = None

        self._check_audio_file()
        logger.info(f"Using device: {self.device}")

    def _check_audio_file(self):
        if not os.path.exists(self.audio_file):
            raise FileNotFoundError(f"Audio file not found: {self.audio_file}")

    def load_model(self):
        logger.info("Loading diarization model...")
        self.model = NeuralDiarizer.from_pretrained(
            model_name=self.model_name,
            vad_model_name=self.vad_model_name,
            map_location=self.device,
            verbose=True
        ).to(self.device)

    def run_diarization(self, batch_size: int = 16):
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Set num_workers to 0 for Windows compatibility
        num_workers = 0 if platform.system() == "Windows" else 4
        
        logger.info("Running diarization...")
        self.annotation = self.model(
            audio_filepath=self.audio_file,
            batch_size=batch_size,
            num_workers=num_workers,
            out_dir=None,
            verbose=True,
        )
        logger.info("✅ Diarization completed!")

    def save_rttm(self):
        if self.annotation is None:
            raise RuntimeError("Diarization not run yet. Call run_diarization() first.")

        stem = os.path.splitext(os.path.basename(self.audio_file))[0]
        self.rttm_file = os.path.join(UPLOAD_FOLDER, f"{stem}.rttm")

        with open(self.rttm_file, "w") as fout:
            self.annotation.write_rttm(fout)

        logger.info(f"RTTM file written to: {self.rttm_file}")
        return self.rttm_file

    def get_speaker_segments(self):
        if self.rttm_file is None or not os.path.exists(self.rttm_file):
            raise RuntimeError("RTTM file not found. Call save_rttm() first.")

        segments = []
        with open(self.rttm_file, "r") as fin:
            for line in fin:
                parts = line.strip().split()
                if len(parts) < 8:
                    continue
                start = float(parts[3])
                duration = float(parts[4])
                speaker = parts[7]
                end = start + duration
                segments.append({
                    "speaker": speaker,
                    "start_time": round(start, 2),
                    "end_time": round(end, 2)
                })
        return segments

# 3. Define routes
@app.route('/')
def home():
    return jsonify({
        "service": "Speaker Diarization API",
        "status": "running",
        "endpoints": {
            "test": "/test (GET)",
            "health": "/health (GET)",
            "diarize": "/diarize (POST)"
        }
    })

@app.route("/health")
def health():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    })

@app.route("/test", methods=["GET"])
def test_endpoint():
    try:
        return jsonify({
            "status": "success",
            "message": "Speaker Diarization API is running",
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "model_loaded": False,
            "timestamp": datetime.utcnow().isoformat()
        })
    except Exception as e:
        logger.error(f"Error in test endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/diarize", methods=["POST"])
def diarize_audio():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    try:
        diarizer = SpeakerDiarizer(audio_file=filepath)
        diarizer.load_model()
        diarizer.run_diarization()
        diarizer.save_rttm()
        segments = diarizer.get_speaker_segments()

        return jsonify({
            "status": "success",
            "timestamp": datetime.utcnow().isoformat(),
            "speaker_segments": segments
        })

    except Exception as e:
        logger.error(f"Error in diarization: {str(e)}")
        return jsonify({"error": str(e)}), 500
    finally:
        # Cleanup uploaded file
        if os.path.exists(filepath):
            os.remove(filepath)

# 4. Run server
if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    logger.info("Starting Flask application...")
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
