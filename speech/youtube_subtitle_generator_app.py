import os

import gradio as gr
import whisper
import torch
from pytubefix import YouTube
from whisper.utils import get_writer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

class WhisperInferencer:
    def __init__(self):
        self.model = whisper.load_model("turbo").to(device)
        self.srt_writer = get_writer(output_format="srt", output_dir=".")

    def inference(self, audio_file_path: str) -> str:
        transcript = self.model.transcribe(audio_file_path)
        self.srt_writer(transcript, audio_file_path)

        filename = os.path.basename(audio_file_path)
        filename = filename.split(".")[0]

        return f"{filename}.srt"

whipser_inferencer = WhisperInferencer()

def transcribe(link: str):
    video_file_name = "video_from_youtube.mp4"
    audio_file_name = "audio_from_youtube.webm"
    yt = YouTube(link)

    # Extract video
    streams = yt.streams.filter(progressive=True, file_extension="mp4", type="video").order_by("resolution").desc()
    streams[0].download(filename=video_file_name)

    # Extract audio
    audio_streams = yt.streams.filter(type="audio").order_by("abr").desc()
    audio_streams[0].download(filename=audio_file_name)

    transcript_file = whipser_inferencer.inference(audio_file_name)
    return transcript_file, [video_file_name, transcript_file]

# Set gradio app
with gr.Blocks() as app:
    gr.Markdown("# Youtube 자막 생성기")

    with gr.Row():
        with gr.Column(scale=1):
            link = gr.Textbox(label="Youtube Link")
            subtitle = gr.File(label="Subtitle", file_types=[".srt"])
            transcribe_btn = gr.Button(value="자막 생성!")

        with gr.Column(scale=4):
            output_video = gr.Video(label="Output", height=500)

    transcribe_btn.click(transcribe, [link], [subtitle, output_video])

# Kubeflow 프록시 경로 설정
KUBEFLOW_BASE_URL = "https://haiqv.ai/notebook/ns-1/l-test/proxy/7864/"

# Gradio 앱 실행
app.launch(
    server_name="0.0.0.0",  # 모든 네트워크 인터페이스에서 접근 가능
    server_port=7864,       # Kubeflow 프록시가 사용하는 포트
    share=False,
    inline=False,
    root_path=KUBEFLOW_BASE_URL
)

app.close()
del whipser_inferencer
torch.cuda.empty_cache()