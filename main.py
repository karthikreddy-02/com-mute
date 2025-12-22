from fastapi import FastAPI, UploadFile, File, Form
from engine import AudioEngine
import os
from dotenv import load_dotenv

app = FastAPI()
load_dotenv()

hf_token = os.getenv('HF_TOKEN')

if not hf_token:
    raise ValueError("HF_TOKEN not found in environment variables!")

engine = AudioEngine(hf_token)

@app.post('/analyze')
async def analyze_audio(file: UploadFile = File(...)):
    path = f"temp_{file.filename}"
    with open(path, "wb") as f:
        f.write(await file.read())
    
    data = engine.process(path)
    return {'segments' : data}

@app.post('/mute')
async def mute_speaker(speaker_id: str = Form(...), file_name: str = Form(...)):
    return {"status": "success", "message": f"Muting {speaker_id}"}


