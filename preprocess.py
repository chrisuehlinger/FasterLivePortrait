import sys
import os
import json
import numpy as np
import cv2
import argparse

# InsightFace detector
try:
    from insightface.app import FaceAnalysis
except ImportError:
    FaceAnalysis = None

# MediaPipe fallback
try:
    import mediapipe as mp
except ImportError:
    mp = None


def detect_face_and_landmarks(img):
    """
    Detects a single face and optionally landmarks in the image.
    Returns an object with .bbox (xmin,ymin,xmax,ymax), .det_score,
    and optionally .landmark_2d_106 (array Nx2).
    """
    h0, w0 = img.shape[:2]
    # 1) Resize shorter side to 1024 for detection
    scale = 1024.0 / min(h0, w0) if min(h0, w0) < 1024 else 1.0
    if scale != 1.0:
        img_det = cv2.resize(img, (int(w0 * scale), int(h0 * scale)))
    else:
        img_det = img

    face = None
    # 2) InsightFace detection
    if FaceAnalysis:
        analyzer = FaceAnalysis(providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        analyzer.prepare(ctx_id=0, det_size=(640, 640))
        faces = analyzer.get(img_det)
        if faces:
            face = faces[0]
    # 3) MediaPipe fallback
    if face is None and mp:
        mp_fd = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.3)
        res = mp_fd.process(cv2.cvtColor(img_det, cv2.COLOR_BGR2RGB))
        if res.detections:
            d = res.detections[0].location_data.relative_bounding_box
            ih, iw = img_det.shape[:2]
            x1 = d.xmin * iw
            y1 = d.ymin * ih
            w_box = d.width * iw
            h_box = d.height * ih
            bbox = np.array([x1, y1, x1 + w_box, y1 + h_box], dtype=float)
            face = type("F", (), {})()
            face.bbox = bbox
            face.det_score = float(res.detections[0].score[0])
            # No 106 landmarks from MediaPipe
            face.landmark_2d_106 = None
    if face is None:
        return None

    # 4) Map bbox back to original image coords
    bbox = face.bbox.astype(float)
    if scale != 1.0:
        bbox /= scale
        face.bbox = bbox
        if getattr(face, 'landmark_2d_106', None) is not None:
            # map landmarks back
            lm = face.landmark_2d_106.astype(float)
            lm /= scale
            face.landmark_2d_106 = lm
    return face


def generate_json_from_image(image_path, output_json_path):
    # Load
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")
    h0, w0 = img.shape[:2]
    # Detect face
    face = detect_face_and_landmarks(img)
    if face is None:
        raise RuntimeError("No face detected by InsightFace or MediaPipe.")

    # Extract bbox and score
    x1, y1, x2, y2 = face.bbox.astype(float)
    width = x2 - x1
    height = y2 - y1
    score = float(getattr(face, 'det_score', 1.0))

    # Crop & align
    scale_factor = 1.25
    y_shift = 0.0
    cx = x1 + width / 2
    cy = y1 + height / 2 + y_shift * height
    size = int(np.ceil(max(width, height) * scale_factor))
    half = size / 2
    xs = int(max(0, cx - half))
    ys = int(max(0, cy - half))
    xe = int(min(w0, cx + half))
    ye = int(min(h0, cy + half))
    crop = img[ys:ye, xs:xe]
    ch, cw = crop.shape[:2]
    pad = np.zeros((size, size, 3), dtype=img.dtype)
    y_off = int(max(0, -(cy - half)))
    x_off = int(max(0, -(cx - half)))
    pad[y_off:y_off+ch, x_off:x_off+cw] = crop
    aligned = cv2.resize(pad, (512, 512))

    # Landmarks on aligned
    lm_list = []
    if getattr(face, 'landmark_2d_106', None) is not None:
        for (lx, ly) in face.landmark_2d_106.astype(float):
            lx_a = (lx - (cx - half)) / size * 512
            ly_a = (ly - (cy - half)) / size * 512
            lm_list.append([float(lx_a), float(ly_a)])

    # Save aligned and landmarked images
    base = os.path.splitext(os.path.basename(image_path))[0]
    out_dir = os.path.dirname(output_json_path) or '.'
    aligned_name = os.path.join(out_dir, f"{base}_aligned.png")
    marked_name = os.path.join(out_dir, f"{base}_landmarked.png")
    cv2.imwrite(aligned_name, aligned)
    marked = aligned.copy()
    for (lx, ly) in lm_list:
        cv2.circle(marked, (int(lx), int(ly)), 2, (0, 0, 255), -1)
    cv2.imwrite(marked_name, marked)

    # Build JSON
    data = {
        "original_image": os.path.basename(image_path),
        "original_size": [w0, h0],
        "detected_bbox": {"x": x1, "y": y1, "width": width, "height": height, "score": score},
        "crop_parameters": {"scale_factor": scale_factor, "vertical_shift": y_shift, "rotation_degree": 0.0, "output_size": [512, 512]},
        "aligned_face_image": os.path.basename(aligned_name),
        "landmarks_aligned": lm_list,
        "flame_model": {"shape": [], "expression": [],
                         "pose": {"yaw": 0.0, "pitch": 0.0, "roll": 0.0, "jaw_open": 0.0},
                         "camera": {"focal_length": 1.0, "principal_point": [0.0, 0.0]},
                         "translation": [0.0, 0.0, 0.0]}
    }
    with open(output_json_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"JSON: {output_json_path}")
    print(f"Aligned: {aligned_name}")
    print(f"Landmarked: {marked_name}")


def main():
    parser = argparse.ArgumentParser(description="Generate FasterLivePortrait preprocessing JSON and images.")
    parser.add_argument("image", help="Input source image path")
    parser.add_argument("output", help="Output JSON path")
    args = parser.parse_args()
    generate_json_from_image(args.image, args.output)


if __name__ == '__main__':
    main()
