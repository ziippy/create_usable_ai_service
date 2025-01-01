from openai import OpenAI

client = OpenAI(api_key="...")

model_name = "whisper-1"

audio_path = "examples/example.wav"
with open(audio_path, "rb") as audio_file:
    transcription = client.audio.transcriptions.create(
        model=model_name,
        file=audio_file,
        response_format="text"
    )

print(transcription)