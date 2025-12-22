import whisper
from pyannote.audio import Pipeline
import torch
from pydub import AudioSegment

class AudioEngine:
    def __init__(self,hf_token):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'Initializing device on {self.device}')

        self.stt_model = whisper.load_model('base')
        self.diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token = hf_token,
        ).to(self.device)

    def process(self, filepath):
        # Transcribe the incoming video with the filepath
        result = self.stt_model.transcribe(filepath)
        segments = result.get("segments",[])

        diarization = self.diarization_pipeline(filepath)

        merged = []

        for segment in segments:
            mid = (segment['start'] + segment['end'])/2
            speaker = 'UNKNOWN'

            for turn, _, current_speaker in diarization.itertracks(yield_label = True):
                if turn.start <= mid <= turn.end:
                    speaker = current_speaker
                    break
            merged.append({
                "start": segment['start'], "end": segment['end'], 
                "text": segment['text'], "speaker": speaker
            }
            )
        
        return merged
    
    def create_muted_file(self, filepath, target_speaker, merged_data):
        audio = AudioSegment.from_file(filepath)

        for data in merged_data:
            if data['speaker'] == target_speaker:
                start_ms, end_ms = int(data['start']*1000), int(data['end'] * 1000)
                silence = AudioSegment.silent(duration = end_ms - start_ms)
        
        out_path = f'muted_{target_speaker}.mp3'
        audio.export(out_path, format = 'mp3')
        return out_path


    
