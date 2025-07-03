# from sahi.slicing import slice_image
# import cv2

# # Load image
# image_path = r"image_data\main_data\small-vehicles1.jpeg"
# image = cv2.imread(image_path)

# # Check if image loaded successfully
# if image is None:
#     raise FileNotFoundError(f"Image not found at {image_path}")

# # Slice the image and save slices as PNGs
# slices = slice_image(
#     image=image,
#     output_dir=r"image_data\data_processing_area",
#     slice_height=256,
#     slice_width=256,
#     overlap_height_ratio=0.2,
#     overlap_width_ratio=0.2,
#     save_as_png=True,  # <-- Add this line
#     output_file_name="slice"  # Optional: custom prefix
# )

# from sahi.slicing import slice_image

# slice_image_result = slice_image(
#     image=image_path,
#     output_file_name=output_file_name,
#     output_dir=output_dir,
#     slice_height=256,
#     slice_width=256,
#     overlap_height_ratio=0.2,
#     overlap_width_ratio=0.2,
# )
# slices is a list of dicts with:
# {
#     "image": np.ndarray (the crop),
#     "start_x": int,
#     "start_y": int,
#     "end_x": int,
#     "end_y": int
# }
from sahi.slicing import slice_image
import cv2
import os

# Load image
image_path = "/home/pi/SAHI-YOLOX/image_data/main_data/small-vehicles1.jpeg"
image = cv2.imread(image_path)

# Check if image loaded successfully
if image is None:
    raise FileNotFoundError(f"Image not found at {image_path}")

# Output directory
output_dir = "/home/pi/SAHI-YOLOX/image_data/data_processing_area"
os.makedirs(output_dir, exist_ok=True)

# Slice the image (returns list of dicts with slice image + metadata)
slices = slice_image(
    image=image,
    slice_height=640,
    slice_width=640,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
)

# Manually save the slices
for idx, slice_info in enumerate(slices):
    slice_img = slice_info["image"]
    save_path = os.path.join(output_dir, f"slice_{idx:03d}.png")
    cv2.imwrite(save_path, slice_img)
    print(f"Saved: {save_path}")
