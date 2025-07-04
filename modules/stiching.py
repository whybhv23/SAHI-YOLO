import re
import math
from PIL import Image, ImageDraw

# === CONFIG ===
ORIG_IMAGE_PATH = "image_data/main_data/city-7569067.jpg"
CHUNK_WIDTH = 640
CHUNK_HEIGHT = 640
DETECTIONS_TXT = "results/all_detections.txt"

# === Load original image ===
orig_img = Image.open(ORIG_IMAGE_PATH)
draw = ImageDraw.Draw(orig_img)

# === Read detection file ===
with open(DETECTIONS_TXT, "r") as f:
    lines = f.readlines()

# Use math.ceil to handle images not perfectly divisible by chunk size
chunks_per_row = math.ceil(orig_img.width / CHUNK_WIDTH)

chunk_index = None
chunk_x_offset = 0
chunk_y_offset = 0

for line in lines:
    line = line.strip()

    # Detect new chunk line, e.g. "? slice_041.png - 4 detections"
    chunk_match = re.match(r"^\?\s*slice_(\d+)\.png", line)
    if chunk_match:
        chunk_index = int(chunk_match.group(1))
        row = chunk_index // chunks_per_row
        col = chunk_index % chunks_per_row
        chunk_x_offset = col * CHUNK_WIDTH
        chunk_y_offset = row * CHUNK_HEIGHT
        continue

    # Detection line: [label] (x1, y1, x2, y2), Score: ...
    det_match = re.match(r"\[(.*?)\]\s+\((\-?\d+),\s*(\-?\d+),\s*(\-?\d+),\s*(\-?\d+)\)", line)
    if det_match and chunk_index is not None:
        label = det_match.group(1)
        x1, y1, x2, y2 = map(int, det_match.groups()[1:])

        # Shift to main image coordinates
        main_x1 = chunk_x_offset + x1
        main_y1 = chunk_y_offset + y1
        main_x2 = chunk_x_offset + x2
        main_y2 = chunk_y_offset + y2

        # Draw box and label
        draw.rectangle([main_x1, main_y1, main_x2, main_y2], outline="green", width=6)
        draw.text((main_x1 + 2, main_y1 - 12), label, fill="green")

# Save result
output_path = "stitched_detections.png"
orig_img.save(output_path)
print(f"[✔] Saved stitched detection image as {output_path}")
