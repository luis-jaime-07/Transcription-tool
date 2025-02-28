import os
import torch
from flask import Flask, request, jsonify, render_template
import whisper
from deep_translator import GoogleTranslator
from pydub import AudioSegment
import datetime

app = Flask(__name__, template_folder='templates')

# Load the 'base' model to reduce memory usage
device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model("base").to(device)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

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
    return str(datetime.timedelta(seconds=int(seconds)))

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

    input_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(input_path)

    # Convert to WAV if necessary
    audio_path = convert_to_wav(input_path)

    # Perform transcription
    result = model.transcribe(audio_path)

    # Format transcription with timestamps
    full_transcription = "\n".join([
        f"[{format_timestamp(segment['start'])}] {segment['text']}"
        for segment in result['segments']
    ])

    detected_language = result.get('language', 'unknown')

    translated_text = None
    if language in ['tl', 'en']:
        translated_text = GoogleTranslator(source=detected_language, target=language).translate(full_transcription)

    return jsonify({
        'detected_language': detected_language,
        'transcription': full_transcription.strip(),
        'translated_text': translated_text
    })

if __name__ == "__main__":
    from waitress import serve
    port = int(os.environ.get("PORT", 10000))  # Use PORT environment variable if available
    serve(app, host="0.0.0.0", port=port)
