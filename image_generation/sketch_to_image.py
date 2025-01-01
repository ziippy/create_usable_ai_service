import os
from typing import IO

import gradio as gr
import requests
import torch
from tqdm import tqdm
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image

WIDTH = 512
HEIGHT = 512

PIPELINE = None

def download_model(url: str) -> str:
    model_id = url.replace("https://civitai.com/models/", "").split("/")[0]

    try:
        response = requests.get(f"https://civitai.com/api/v1/models/{model_id}", timeout=600)
    except Exception as err:
        print(f"[ERROR] {err}")
        raise err

    download_url = response.json()["modelVersions"][0]["downloadUrl"]
    filename = response.json()["modelVersions"][0]["files"][0]["name"]

    file_path = f"models/{filename}"
    if os.path.exists(file_path):
        print(f"[INFO] File already exists: {file_path}")
        return file_path

    os.makedirs("models", exist_ok=True)
    download_from_url(download_url, file_path)
    print(f"[INFO] File downloaded: {file_path}")
    return file_path

def download_from_url(url: str, file_path: str, chunk_size=1024):
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get('content-length', 0))
    with open(file_path, 'wb') as file, tqdm(
        desc=file_path,
        total=total,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)

def init_pipeline(model_file: IO) -> str:
    print("[INFO] Initialize pipeline")
    global PIPELINE
    PIPELINE = StableDiffusionImg2ImgPipeline.from_single_file(
        model_file.name,
        torch_dtype=torch.float16,
        use_safetensors=True,
    ).to("cuda")
    print("[INFO] Initialized pipeline")
    return "Model Loaded!"

def sketch_to_image(sketch: Image.Image, prompt: str, negative_prompt: str):
    width, height = sketch.size
    images =  PIPELINE(
        image=sketch,
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_images_per_prompt=4,
        num_inference_steps=20,
        strength=0.7,
    ).images

    with torch.cuda.device("cuda"):
        torch.cuda.empty_cache()

    return images

# 24.12.02 모델 다운로드 UI 이슈
# 내부 버전 이슈로 모델 다운로드가 UI 상에서 정상적으로 동작하지 않습니다.
# 아래 코드로 UI에서 모델을 다운로드하는 방식이 아닌 코드 상에서 바로 다운로드 하시기 바랍니다.
# 이후 UI에서 같은 url을 입력 후 모델 다운로드 버튼을 누르면 미리 설치한 모델을 불러오게 됩니다.
url = "https://civitai.com/models/65203/disney-pixar-cartoon-type-a"
model_path = download_model(url)
print("model_path", model_path)

print("[INFO] Gradio app ready")
with gr.Blocks() as app:
    gr.Markdown("# 스케치 to 이미지 애플리케이션")

    gr.Markdown("## 모델 다운로드")
    with gr.Row():
        model_url = gr.Textbox(label="Model Link", placeholder="https://civitai.com/")
        download_model_btn = gr.Button(value="Download model")
    with gr.Row():
        model_file = gr.File(label="Model File")

    gr.Markdown("## 모델 불러오기")
    with gr.Row():
        load_model_btn = gr.Button(value="Load model")
    with gr.Row():
        is_model_check = gr.Textbox(label="Model Load Check", value="Model Not loaded")

    gr.Markdown("## 프롬프트 입력")
    with gr.Row():
        prompt = gr.Textbox(label="Prompt")
    with gr.Row():
        n_prompt = gr.Textbox(label="Negative Prompt")

    gr.Markdown("## 스케치 to 이미지 생성")
    with gr.Row():
        with gr.Column():
            with gr.Tab("Canvas"):
                with gr.Row():
                    canvas = gr.Image(
                        label="Draw",
                        # sources=["canvas"],
                        image_mode="RGB",
                        # tool="color-sketch",
                        interactive=True,
                        width=WIDTH,
                        height=HEIGHT,
                        # shape=(WIDTH, HEIGHT),
                        # brush_radius=20,
                        type="pil",
                    )
                with gr.Row():
                    canvas_run_btn = gr.Button(value="Generate")

            with gr.Tab("File"):
                with gr.Row():
                    file = gr.Image(
                        label="Upload",
                        sources=["upload"],
                        image_mode="RGB",
                        # tool="color-sketch",
                        interactive=True,
                        width=WIDTH,
                        height=HEIGHT,
                        # shape=(WIDTH, HEIGHT),
                        type="pil",
                    )
                with gr.Row():
                    file_run_btn = gr.Button(value="Generate")

        with gr.Column():
            result_gallery = gr.Gallery(label="Output", height=512)


    # Event
    download_model_btn.click(
        download_model,
        [model_url],
        [model_file],
    )
    load_model_btn.click(
        init_pipeline,
        [model_file],
        [is_model_check],
    )
    canvas_run_btn.click(
        sketch_to_image,
        [canvas, prompt, n_prompt],
        [result_gallery],
    )
    file_run_btn.click(
        sketch_to_image,
        [file, prompt, n_prompt],
        [result_gallery],
    )

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