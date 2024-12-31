import os
import urllib
from typing import Tuple

import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# Implement inferencer
CHECKPOINT_PATH = os.path.join("checkpoint")
CHECKPOINT_NAME = "sam_vit_h_4b8939.pth"
CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
MODEL_TYPE = "default"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SAMInferencer:
    def __init__(
        self,
        checkpoint_path: str,
        checkpoint_name: str,
        checkpoint_url: str,
        model_type: str,
        device: torch.device,
    ):
        print("[INFO] Initailize inferencer")
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path, exist_ok=True)
        checkpoint = os.path.join(checkpoint_path, checkpoint_name)
        if not os.path.exists(checkpoint):
            urllib.request.urlretrieve(checkpoint_url, checkpoint)
        sam = sam_model_registry[model_type](checkpoint=checkpoint).to(device)
        self.predictor = SamPredictor(sam)

    def inference(
        self,
        image: np.ndarray,
        point_coords: np.ndarray,
        points_labels: np.ndarray,
    ) -> np.ndarray:
        self.predictor.set_image(image)
        masks, scores, _ = self.predictor.predict(point_coords, points_labels)
        mask, _ = self.select_mask(masks, scores)
        return mask

    def select_mask(
        self, masks: np.ndarray, scores: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Determine if we should return the multiclick mask or not from the number of points.
        # The reweighting is used to avoid control flow.
        # Reference: https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/utils/onnx.py#L92-L105
        score_reweight = np.array([-1000] + [0] * 2)
        score = scores + score_reweight
        best_idx = np.argmax(score)
        selected_mask = np.expand_dims(masks[best_idx, :, :], axis=-1)
        selected_score = np.expand_dims(scores[best_idx], axis=0)
        return selected_mask, selected_score


inferencer = SAMInferencer(
    CHECKPOINT_PATH, CHECKPOINT_NAME, CHECKPOINT_URL, MODEL_TYPE, DEVICE
)

# Implement event function
def extract_object(image: np.ndarray, point_x: int, point_y: int):
    point_coords = np.array([[point_x, point_y], [0, 0]])
    point_label = np.array([1, -1])

    # Get mask
    mask = inferencer.inference(image, point_coords, point_label)

    # Extract object and remove background
    # Postprocess mask
    mask = (mask > 0).astype(np.uint8)

    # Remove background
    result_image = cv2.bitwise_and(image, image, mask=mask)

    # Convert to rgba channel
    bgr_channel = result_image[..., :3]  # BGR 채널 분리
    alpha_channel = np.where(bgr_channel[..., 0] == 0, 0, 255).astype(np.uint8)
    result_image = np.dstack((bgr_channel, alpha_channel))  # BGRA 이미지 생성

    return result_image


def extract_object_by_event(image: np.ndarray, evt: gr.SelectData):
    click_x, click_y = evt.index

    return extract_object(image, click_x, click_y)


def get_coords(evt: gr.SelectData):
    return evt.index[0], evt.index[1]


# Implement app
with gr.Blocks() as app:
    gr.Markdown("# Interactive Remove Background from Image")
    with gr.Row():
        coord_x = gr.Number(label="Mouse coords x")
        coord_y = gr.Number(label="Mouse coords y")

    with gr.Row():
        input_img = gr.Image(label="Input image", height=600)
        output_img = gr.Image(label="Output image", height=600)

    input_img.select(get_coords, None, [coord_x, coord_y])
    input_img.select(extract_object_by_event, [input_img], output_img)

    gr.Markdown("## Image Examples")
    gr.Examples(
        examples=[
            [os.path.join(os.getcwd(), "examples/dog.jpg"), 1013, 786],
            [os.path.join(os.getcwd(), "examples/mannequin.jpg"), 1720, 230],
        ],
        inputs=[input_img, coord_x, coord_y],
        outputs=output_img,
        fn=extract_object,
        run_on_click=True,
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