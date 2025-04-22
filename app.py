import os
import torch
import numpy as np
from flask import Flask, request, jsonify
from nemo.collections.asr.models.msdd_models import NeuralDiarizer
from werkzeug.utils import secure_filename

# Back‑port sctypes for backward compatibility
np.sctypes = {
    'int': [np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64],
    'uint': [np.uint8, np.uint16, np.uint32, np.uint64, np.uint],
    'float': [np.float16, np.float32, np.float64],
    'complex': [np.complex64, np.complex128]
}

# 1. Init Flask app
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "diarization_output"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# 2. Load model once (when server starts)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = NeuralDiarizer.from_pretrained(
    model_name="diar_msdd_telephonic",
    vad_model_name="vad_multilingual_marblenet",
    map_location=device,
    verbose=True,
).to(device)

# 3. Define routes
@app.route('/test', methods=['GET'])
def test_app():
    return jsonify({'message': 'Server is up and running 🚀'}), 200

@app.route('/diarize', methods=['POST'])
def diarize_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save uploaded audio
    filename = secure_filename(audio_file.filename)
    audio_path = os.path.join(UPLOAD_FOLDER, filename)
    audio_file.save(audio_path)

    # Run diarization
    annotation = model(
        audio_filepath=audio_path,
        batch_size=16,
        num_workers=4,
        out_dir=OUTPUT_FOLDER,
        verbose=False,
    )

    # Process output
    results = []
    for segment in annotation.to_list():
        start = segment['start']
        end = segment['end']
        speaker = segment['label']
        results.append({
            'speaker': speaker,
            'start': round(start, 2),
            'end': round(end, 2)
        })

    return jsonify(results)

# 4. Run server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
