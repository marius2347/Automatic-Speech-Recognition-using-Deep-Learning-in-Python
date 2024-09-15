# Automatic Speech Recognition using Deep Learning in Python developed by Marius

# Run the app with: `uvicorn filename:app --reload`

# libraries
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import torch
import librosa
import io

# FastAPI app
app = FastAPI()

# pre-trained Wav2Vec2 model and tokenizer
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")

# reads the audio file and preprocesses it
def load_audio(file_bytes):
    waveform, sample_rate = librosa.load(io.BytesIO(file_bytes), sr=16000)
    return waveform

# function takes the audio data and returns the transcription
def transcribe_audio(audio_bytes):
    audio = load_audio(audio_bytes)
    inputs = tokenizer(audio, return_tensors="pt", padding="longest")
    with torch.no_grad():
        logits = model(inputs.input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = tokenizer.batch_decode(predicted_ids)
    
    # Convert transcription to sentence case or lowercase
    transcription_text = transcription[0].capitalize()  #
    
    return transcription_text + "."

# the HTML form with CSS and JavaScript
@app.get("/", response_class=HTMLResponse)
async def get_form():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Audio Transcription</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background-color: #f0f0f0;
                margin: 0;
                padding: 0;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
            }
            .container {
                background-color: #ffffff;
                border-radius: 8px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                padding: 20px;
                width: 90%;
                max-width: 500px;
                text-align: center;
            }
            input[type="file"] {
                margin-bottom: 20px;
            }
            input[type="submit"] {
                background-color: #007bff;
                color: #ffffff;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
            }
            input[type="submit"]:hover {
                background-color: #0056b3;
            }
            #transcription {
                margin-top: 20px;
                font-size: 18px;
                color: #333;
                text-align: left;
                white-space: pre-wrap;
                word-wrap: break-word;
            }
            audio {
                margin-top: 20px;
                width: 100%;
                max-width: auto;
            }
            
            @media (max-width: 600px) {
                .container {
                    width: 90%;
                    padding: 10px;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h2 style = "color: #ff5733">Upload an MP3 File for Transcription</h2>
            <form id="uploadForm" enctype="multipart/form-data">
                <input type="file" id="fileInput" name="file" accept=".mp3" required>
                <input type="submit" value="Upload">
            </form>
            <div id="transcription">Upload a file to see the transcription here...</div>
            <audio id="audioPlayer" controls style="display: none;">
                <source id="audioSource" type="audio/mpeg">
                Your browser does not support the audio element.
            </audio>
        </div>
        <script>
            document.getElementById('uploadForm').addEventListener('submit', async (event) => {
            event.preventDefault();
            
            const fileInput = document.getElementById('fileInput');
            const file = fileInput.files[0];
            
            if (!file) {
                alert('Please select a file');
                return;
            }
            
            const formData = new FormData();
            formData.append('file', file);
            
            try {
                const response = await fetch('/transcribe/', {
                    method: 'POST',
                    body: formData
                });
                const result = await response.json();  // Get plain transcription text
                
                // initialize the transcription area
                const transcriptionElement = document.getElementById('transcription');
                transcriptionElement.innerHTML = '';  // Clear previous transcription

                // display the transcription with a typing effect
                typeEffect(result, transcriptionElement);

                // play the audio
                const audioPlayer = document.getElementById('audioPlayer');
                const audioSource = document.getElementById('audioSource');
                audioSource.src = URL.createObjectURL(file);
                audioPlayer.style.display = 'block';
                audioPlayer.load();
                
            } catch (error) {
                document.getElementById('transcription').innerHTML = 'An error occurred. Please try again.';
            }
        });

            // typing effect function
            function typeEffect(text, element) {
                let index = 0;
                const speed = 50; // milliseconds per character

                function type() {
                    if (index < text.length) {
                        element.innerHTML += text.charAt(index);  // add one character at a time
                        index++;
                        setTimeout(type, speed);  // delay before typing the next character
                    }
                }
                type();
            }
    </script>
    </body>
    </html>
    """

# the file upload endpoint
@app.post("/transcribe/")
async def transcribe(file: UploadFile = File(...)):
    if file.content_type != "audio/mpeg":
        return HTMLResponse(content="Error: Invalid file type. Please upload an mp3 file.", status_code=400)
    
    # read the file content
    file_bytes = await file.read()

    try:
        # get the transcription
        transcription = transcribe_audio(file_bytes)
        return JSONResponse(content=transcription)  
    except Exception as e:
        return JSONResponse(content=f"Error: {str(e)}", status_code=500)
