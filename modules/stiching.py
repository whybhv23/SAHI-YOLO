# import re
# import os
# from PIL import Image, ImageDraw

# === CONFIG ===
# ORIG_IMAGE_PATH = "image_data/main_data/small-vehicles1.jpeg"
# SLICES_DIR = "image_data/data_processing_area"
# DETECTIONS_TXT = "text_data/results.txt"
# CHUNK_WIDTH = 640
# CHUNK_HEIGHT = 580

# # === Load original image ===
# orig_img = Image.open(ORIG_IMAGE_PATH)
# draw = ImageDraw.Draw(orig_img)

# # === Read detections file ===
# with open(DETECTIONS_TXT, "r") as f:
#     lines = f.readlines()

# current_chunk_name = ""
# chunk_detections = []

# # Process each line
# for line in lines:
#     line = line.strip()
    
#     if line.startswith("?slice_"):
#         # New chunk
#         current_chunk_name = line.split()[0][1:]  # remove "?"
#         chunk_index = int(re.findall(r"\d+", current_chunk_name)[0])
#         print(f"Processing chunk: {current_chunk_name} (index {chunk_index})")
        
#         # Compute chunk position
#         num_chunks_per_row = orig_img.width // CHUNK_WIDTH
#         row = chunk_index // num_chunks_per_row
#         col = chunk_index % num_chunks_per_row
#         chunk_x_offset = col * CHUNK_WIDTH
#         chunk_y_offset = row * CHUNK_HEIGHT
#         print(f"Chunk position: ({chunk_x_offset}, {chunk_y_offset})")
        
#     elif line.startswith("["):
#         # Detection line
#         match = re.search(r"\((\d+),(\d+),(\d+),(\d+)\)", line)
#         if match:
#             x1, y1, x2, y2 = map(int, match.groups())
            
#             # Shift to main image
#             main_x1 = chunk_x_offset + x1
#             main_y1 = chunk_y_offset + y1
#             main_x2 = chunk_x_offset + x2
#             main_y2 = chunk_y_offset + y2
#             print(f"Detection: ({main_x1}, {main_y1}, {main_x2}, {main_y2})")

            
#             # Draw on original image
#             draw.rectangle([main_x1, main_y1, main_x2, main_y2], outline="red", width=2)

# # === Save output ===
# orig_img.save("stitched_detections.png")
# print("Stitched image saved as stitched_detections.png")



import os
import re
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

# === Read detection file ===
with open(DETECTIONS_TXT, "r") as f:
    lines = f.readlines()

chunk_index = -1
chunk_x_offset = 0
chunk_y_offset = 0

# Calculate chunks per row from original image width
chunks_per_row = orig_img.width // CHUNK_WIDTH

for line in lines:
    line = line.strip()

    # Detect new chunk line
    chunk_match = re.match(r"^\??\s*slice_(\d+)\.png", line)
    if chunk_match:
        chunk_index = int(chunk_match.group(1))
        row = chunk_index // chunks_per_row
        col = chunk_index % chunks_per_row
        chunk_x_offset = col * CHUNK_WIDTH
        chunk_y_offset = row * CHUNK_HEIGHT
        continue

    # Detection line
    bbox_match = re.search(r"\[(.*?)\]\s+\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)", line)
    if bbox_match:
        label = bbox_match.group(1)
        x1, y1, x2, y2 = map(int, bbox_match.groups()[1:])

        # Shift to main image coordinates
        main_x1 = chunk_x_offset + x1
        main_y1 = chunk_y_offset + y1
        main_x2 = chunk_x_offset + x2
        main_y2 = chunk_y_offset + y2

        # Draw box and label
        draw.rectangle([main_x1, main_y1, main_x2, main_y2], outline="red", width=2)
        draw.text((main_x1 + 2, main_y1 - 12), label, fill="red")

# Save result
output_path = "stitched_detections.png"
orig_img.save(output_path)
print(f"[✔] Saved stitched detection image as {output_path}")
