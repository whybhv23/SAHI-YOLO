import os
import re
from PIL import Image
import numpy as np
import cv2
from sahi.utils.object_prediction import ObjectPrediction
from sahi.postprocess.combine import merge_object_prediction_list

from sahi.utils.cv import visualize_object_predictions

# --- CONFIG ---
SLICE_DIR = "image_data/data_processing_area"  # update this
ORIG_W, ORIG_H = 1068, 580  # original full image shape (W x H) before slicing
SLICE_W, SLICE_H = 640, 640
OVERLAP_RATIO = 0.2
PREDICTION_TEXT_PATH = "text_data/results.txt"  # your .txt file with the long detection text
OUTPUT_IMAGE_PATH = "stitched_result.png"

# --- HELPERS ---
def parse_prediction_text(prediction_text):
    detections_dict = {}
    current_file = None
    for line in prediction_text.splitlines():
        if line.startswith("?"):
            current_file = line.split("?")[1].strip().split(" - ")[0]
            detections_dict[current_file] = []
        elif "[" in line and "]" in line:
            match = re.match(r"\s+\[(.*?)\] \((\d+), (\d+), (\d+), (\d+)\), Score: ([0-9.]+)", line)
            if match:
                class_name = match.group(1)
                x1, y1, x2, y2 = map(int, match.group(2, 3, 4, 5))
                score = float(match.group(6))
                detections_dict[current_file].append((class_name, (x1, y1, x2, y2), score))
    return detections_dict

def get_slice_grid(orig_w, orig_h, slice_w, slice_h, overlap_ratio):
    step_w = int(slice_w * (1 - overlap_ratio))
    step_h = int(slice_h * (1 - overlap_ratio))
    cols = (orig_w - slice_w) // step_w + 1
    rows = (orig_h - slice_h) // step_h + 1
    return step_w, step_h, cols, rows

def build_object_predictions(detections_dict, orig_shape, slice_size, overlap_ratio):
    orig_h, orig_w = orig_shape
    slice_w, slice_h = slice_size
    step_w, step_h, cols, _ = get_slice_grid(orig_w, orig_h, slice_w, slice_h, overlap_ratio)
    
    predictions = []
    for idx, (filename, detections) in enumerate(sorted(detections_dict.items())):
        row = idx // cols
        col = idx % cols
        shift_x = col * step_w
        shift_y = row * step_h

        for class_name, bbox, score in detections:
            predictions.append(ObjectPrediction(
                bbox=list(bbox),
                category_id=0,
                score=score,
                category_name=class_name,
                shift_amount=[shift_x, shift_y],
                full_shape=(orig_h, orig_w)
            ))
    return predictions

def stitch_slices(slice_dir, orig_shape, slice_size, overlap_ratio):
    orig_h, orig_w = orig_shape
    slice_w, slice_h = slice_size
    step_w, step_h, cols, rows = get_slice_grid(orig_w, orig_h, slice_w, slice_h, overlap_ratio)
    
    canvas = Image.new("RGB", (orig_w, orig_h))
    slice_images = sorted([f for f in os.listdir(slice_dir) if f.endswith(".png")])
    for idx, fname in enumerate(slice_images):
        slice_img = Image.open(os.path.join(slice_dir, fname))
        row = idx // cols
        col = idx % cols
        offset_x = col * step_w
        offset_y = row * step_h
        canvas.paste(slice_img, (offset_x, offset_y))
    return np.array(canvas)

# --- MAIN ---
if __name__ == "__main__":
    with open(PREDICTION_TEXT_PATH, "r", encoding="utf-8") as f:
        prediction_text = f.read()

    detections_dict = parse_prediction_text(prediction_text)
    predictions = build_object_predictions(
        detections_dict,
        orig_shape=(ORIG_H, ORIG_W),
        slice_size=(SLICE_W, SLICE_H),
        overlap_ratio=OVERLAP_RATIO
    )

    merged_predictions = merge_object_prediction_list(
        predictions,
        match_threshold=0.5,
        match_metric="ios",
        class_agnostic=False
    )

    stitched_image = stitch_slices(SLICE_DIR, (ORIG_H, ORIG_W), (SLICE_W, SLICE_H), OVERLAP_RATIO)
    final = visualize_object_predictions(stitched_image, merged_predictions)

    cv2.imwrite(OUTPUT_IMAGE_PATH, final)
    print(f"[✅] Stitched image saved at: {OUTPUT_IMAGE_PATH}")
