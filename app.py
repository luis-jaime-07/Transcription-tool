import os
import torch
from flask import Flask, request, jsonify, render_template
import whisper
from deep_translator import GoogleTranslator
from pydub import AudioSegment
import datetime

app = Flask(__name__, template_folder='templates')

# Check for GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model("small").to(device)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB limit (adjust as needed)

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def convert_to_wav(input_path):
    """Convert any audio format to WAV"""
    output_path = input_path.rsplit(".", 1)[0] + ".wav"
    if input_path.endswith(".wav"):
        return input_path
    audio = AudioSegment.from_file(input_path)
    audio.export(output_path, format="wav")
    return output_path

def format_timestamp(seconds):
    """Convert seconds to HH:MM:SS format"""
    return str(datetime.timedelta(seconds=int(seconds))) if seconds else "00:00:00"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'audio' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['audio']
    language = request.form.get('language', 'none')

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Ensure file size is within limit
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)  # Reset file pointer
    if file_size > MAX_FILE_SIZE:
        return jsonify({"error": "File too large. Max size is 10MB"}), 400

    input_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(input_path)

    # Convert to WAV if necessary
    audio_path = convert_to_wav(input_path)

    # Perform transcription
    result = model.transcribe(audio_path)

    # Format transcription with timestamps
    full_transcription = "\n".join([
        f"[{format_timestamp(segment.get('start'))}] {segment.get('text', '')}"
        for segment in result.get('segments', [])
    ])

    detected_language = result.get('language', 'unknown')

    translated_text = None
    supported_languages = ['tl', 'en']
    if language in supported_languages and detected_language in supported_languages:
        translated_text = GoogleTranslator(source=detected_language, target=language).translate(full_transcription)

    return jsonify({
        'detected_language': detected_language,
        'transcription': full_transcription.strip(),
        'translated_text': translated_text
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Use PORT from environment variables (for Vercel)
    app.run(host="0.0.0.0", port=port, debug=False)
