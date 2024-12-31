import os
import torch
import gradio as gr
import numpy as np
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# Implement inferencer
class TrOCRInferencer:
    def __init__(self):
        print("[INFO] Initialize TrOCR Inferencer.")
        self.processor = TrOCRProcessor.from_pretrained(
            "microsoft/trocr-base-handwritten"
        )
        self.model = VisionEncoderDecoderModel.from_pretrained(
            "microsoft/trocr-base-handwritten"
        ).to(device)

    def inference(self, image: Image) -> str:
        """Inference using model.

        It is performed as a procedure of preprocessing - inference - postprocessing.
        """
        # preprocess
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.to(device)
        # inference
        with torch.no_grad():
            generated_ids = self.model.generate(pixel_values)
        # postprocess
        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]

        return generated_text

inferencer = TrOCRInferencer()


# Implement event function
def image_to_text(image: np.ndarray) -> str:
    image = Image.fromarray(image).convert("RGB")
    text = inferencer.inference(image)
    return text

def proxy_url(path):
    # base_proxy_url = "notebook/ns-1/l-test/proxy/7860/file="
    base_proxy_url = ""
    return base_proxy_url + path

# Implement app
with gr.Blocks() as app:
    gr.Markdown("# Handwritten Image OCR")
    with gr.Tab("Image upload"):
        image = gr.Image(label="Handwritten image file")
        output = gr.Textbox(label="Output Box")
        convert_btn = gr.Button("Convert")
        convert_btn.click(
            fn=image_to_text, inputs=image, outputs=output
        )

        # gr.Markdown("## Image Examples")
        # gr.Examples(
        #     examples=[
        #         proxy_url(os.path.join(os.getcwd(), "examples/Hello.png")),
        #         proxy_url(os.path.join(os.getcwd(), "examples/Hello_cursive.png")),
        #         proxy_url(os.path.join(os.getcwd(), "examples/Red.png")),
        #         proxy_url(os.path.join(os.getcwd(), "examples/sentence.png")),
        #         proxy_url(os.path.join(os.getcwd(), "examples/i_love_you.png")),
        #         proxy_url(os.path.join(os.getcwd(), "examples/merrychristmas.png")),
        #         proxy_url(os.path.join(os.getcwd(), "examples/Rock.png")),
        #         proxy_url(os.path.join(os.getcwd(), "examples/Bob.png")),
        #     ],
        #     inputs=image,
        #     outputs=output,
        #     fn=image_to_text,
        # )

    with gr.Tab("Drawing"):
        sketchpad = gr.Sketchpad(
            label="Handwritten Sketchpad",
            # shape=(600, 192),
            width=600,  # 이전 shape의 가로 크기
            height=192, # 이전 shape의 세로 크기
            # brush_radius=2,
            # invert_colors=False,
        )
        output = gr.Textbox(label="Output Box")
        convert_btn = gr.Button("Convert")
        convert_btn.click(
            fn=image_to_text, inputs=sketchpad, outputs=output
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