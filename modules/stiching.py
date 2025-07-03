import re
import os
from PIL import Image, ImageDraw

# === CONFIG ===
ORIG_IMAGE_PATH = "image_data/main_data/small-vehicles1.jpeg"
SLICES_DIR = "image_data/data_processing_area"
DETECTIONS_TXT = "text_data/results.txt"
CHUNK_WIDTH = 640
CHUNK_HEIGHT = 580

# === Load original image ===
orig_img = Image.open(ORIG_IMAGE_PATH)
draw = ImageDraw.Draw(orig_img)

# === Read detections file ===
with open(DETECTIONS_TXT, "r") as f:
    lines = f.readlines()

current_chunk_name = ""
chunk_detections = []

# Process each line
for line in lines:
    line = line.strip()
    
    if line.startswith("?slice_"):
        # New chunk
        current_chunk_name = line.split()[0][1:]  # remove "?"
        chunk_index = int(re.findall(r"\d+", current_chunk_name)[0])
        
        # Compute chunk position
        num_chunks_per_row = orig_img.width // CHUNK_WIDTH
        row = chunk_index // num_chunks_per_row
        col = chunk_index % num_chunks_per_row
        chunk_x_offset = col * CHUNK_WIDTH
        chunk_y_offset = row * CHUNK_HEIGHT
        
    elif line.startswith("["):
        # Detection line
        match = re.search(r"\((\d+),(\d+),(\d+),(\d+)\)", line)
        if match:
            x1, y1, x2, y2 = map(int, match.groups())
            
            # Shift to main image
            main_x1 = chunk_x_offset + x1
            main_y1 = chunk_y_offset + y1
            main_x2 = chunk_x_offset + x2
            main_y2 = chunk_y_offset + y2
            
            # Draw on original image
            draw.rectangle([main_x1, main_y1, main_x2, main_y2], outline="red", width=2)

# === Save output ===
orig_img.save("stitched_detections.png")
print("Stitched image saved as stitched_detections.png")
