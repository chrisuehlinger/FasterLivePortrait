import gradio as gr
import os
import json
import numpy as np
from PIL import Image, ImageDraw

try:
    from insightface.app import FaceAnalysis
except ImportError:
    raise ImportError("Please install insightface: pip install insightface")

face_analyzer = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
face_analyzer.prepare(ctx_id=0, det_size=(640, 640))

def load_and_display_json(json_file):
    if json_file is None:
        return None, "No JSON file provided."
    with open(json_file.name, 'r') as f:
        data = json.load(f)

    img_path = os.path.join(os.path.dirname(json_file.name), data["aligned_face_image"])
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    for pt in data.get("landmarks_aligned", []):
        x, y = pt
        r = 2
        draw.ellipse([x - r, y - r, x + r, y + r], fill=(255, 0, 0))

    return img, json.dumps(data, indent=2)

def detect_and_generate_json(input_image):
    if input_image is None:
        return None, None, "No image provided."

    img = np.array(input_image)
    faces = face_analyzer.get(img)
    if not faces:
        return None, None, "No face detected."

    face = faces[0]
    bbox = face.bbox.astype(float)
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    det_score = float(getattr(face, 'det_score', 0.0))

    scale_factor = 1.25
    vertical_shift = 0.0
    rotation = 0.0
    center_x = x1 + width / 2
    center_y = y1 + height / 2 + vertical_shift * height
    crop_size = max(width, height) * scale_factor
    half = crop_size / 2
    x_start = center_x - half
    y_start = center_y - half
    x_end = x_start + crop_size
    y_end = y_start + crop_size

    h, w = img.shape[:2]
    xs_i, ys_i = int(max(0, x_start)), int(max(0, y_start))
    xe_i, ye_i = int(min(w, x_end)), int(min(h, y_end))
    crop_img = img[ys_i:ye_i, xs_i:xe_i]
    ch, cw = crop_img.shape[:2]
    if (ch != int(crop_size)) or (cw != int(crop_size)):
        pad_img = np.zeros((int(crop_size), int(crop_size), 3), dtype=img.dtype)
        y_off = int(max(0, -y_start))
        x_off = int(max(0, -x_start))
        pad_img[y_off:y_off+ch, x_off:x_off+cw] = crop_img
        crop_img = pad_img
    aligned = cv2.resize(crop_img, (512, 512))
    aligned_pil = Image.fromarray(cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB))

    landmarks = getattr(face, 'landmark_2d_106', None)
    landmarks_list = []
    if landmarks is not None:
        for (lx, ly) in landmarks.astype(float):
            lx_a = (lx - x_start) / crop_size * 512
            ly_a = (ly - y_start) / crop_size * 512
            landmarks_list.append([float(lx_a), float(ly_a)])

    data = {
        "original_image": "uploaded_image",
        "original_size": [w, h],
        "detected_bbox": {
            "x": float(x1), "y": float(y1),
            "width": float(width), "height": float(height),
            "score": det_score
        },
        "crop_parameters": {
            "scale_factor": scale_factor,
            "vertical_shift": vertical_shift,
            "rotation_degree": rotation,
            "output_size": [512, 512]
        },
        "aligned_face_image": "inline",
        "landmarks_aligned": landmarks_list,
        "flame_model": {
            "shape": [],
            "expression": [],
            "pose": {"yaw": 0.0, "pitch": 0.0, "roll": 0.0, "jaw_open": 0.0},
            "camera": {"focal_length": 1.0, "principal_point": [0.0, 0.0]},
            "translation": [0.0, 0.0, 0.0]
        }
    }
    return aligned_pil, json.dumps(data, indent=2), "Success."

with gr.Blocks() as demo:
    gr.Markdown("## FasterLivePortrait JSON Editor and Generator")

    with gr.Tab("Load and View JSON"):
        json_input = gr.File(label="Upload JSON File")
        json_output_img = gr.Image(label="Face Image with Landmarks")
        json_output_text = gr.Textbox(label="JSON Content", lines=20)
        json_input.change(fn=load_and_display_json, inputs=json_input,
                          outputs=[json_output_img, json_output_text])

    with gr.Tab("Detect and Generate JSON"):
        img_input = gr.Image(type="pil", label="Upload Image")
        out_img = gr.Image(label="Aligned Face")
        out_json = gr.Textbox(label="Generated JSON", lines=20)
        status = gr.Textbox(label="Status")
        img_input.change(fn=detect_and_generate_json, inputs=img_input,
                         outputs=[out_img, out_json, status])

if __name__ == "__main__":
    demo.launch()