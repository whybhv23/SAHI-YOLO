import os
import cv2
import argparse
from picamera2.devices import Hailo

def extract_detections(hailo_output, w, h, class_names, threshold=0.5):
    """Extract detections from the HailoRT-postprocess output."""
    results = []
    for class_id, detections in enumerate(hailo_output):
        for detection in detections:
            score = detection[4]
            if score >= threshold:
                y0, x0, y1, x1 = detection[:4]
                bbox = (int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h))
                results.append([class_names[class_id], bbox, score])
    return results

# /home/pi/SAHI-YOLOX/yolov8n.hef
# /home/pi/SAHI-YOLOX/labels.txt
# /home/pi/SAHI-YOLOX/image_data/data_processing_area

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Hailo inference on SAHI-sliced images")
    parser.add_argument("-m", "--model", default="/home/pi/SAHI-YOLOX/yolov8n.hef", help="Path to Hailo model file")
    parser.add_argument("-l", "--labels", default="/home/pi/SAHI-YOLOX/labels.txt", help="Path to class labels file")
    parser.add_argument("-i", "--input_dir", default="/home/pi/SAHI-YOLOX/image_data/data_processing_area",help="Directory containing input images")
    parser.add_argument("-s", "--score_thresh", type=float, default=0.5, help="Confidence threshold")
    args = parser.parse_args()

    # Load class labels
    with open(args.labels, 'r', encoding="utf-8") as f:
        class_names = f.read().splitlines()

    # Load model using Hailo API
    with Hailo(args.model) as hailo:
        input_h, input_w, _ = hailo.get_input_shape()
        print(f"Model input shape: {input_w}x{input_h}")

        os.makedirs("results", exist_ok=True)  # Make sure results dir exists
        all_txt_path = os.path.join("results", "all_detections.txt")

        for filename in sorted(os.listdir(args.input_dir)):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                img_path = os.path.join(args.input_dir, filename)
                image = cv2.imread(img_path)

                if image is None:
                    print(f"? Failed to read: {img_path}")
                    with open(all_txt_path, "a", encoding="utf-8") as all_f:
                        all_f.write(f"? Failed to read: {img_path}\n")
                    continue

                # Resize to model input
                resized = cv2.resize(image, (input_w, input_h))
                output = hailo.run(resized)

                detections = extract_detections(output, image.shape[1], image.shape[0], class_names, args.score_thresh)

                header = f"\n? {filename} - {len(detections)} detections"
                print(header)
                with open(all_txt_path, "a", encoding="utf-8") as all_f:
                    all_f.write(header + "\n")
                    for class_name, bbox, score in detections:
                        det_str = f"  [{class_name}] {bbox}, Score: {score:.2f}"
                        print(det_str)
                        all_f.write(det_str + "\n")

                # txt_path = os.path.join("results", f"{os.path.splitext(filename)[0]}.txt")
                # with open(txt_path, "w", encoding="utf-8") as f:
                #     for class_name, bbox, score in detections:
                #         f.write(f"{class_name} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]} {score:.4f}\n")

                # # Optional: draw and save
                # for class_name, bbox, score in detections:
                #     cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255,0, 255), 2)
                #     cv2.putText(image, f"{class_name} {int(score * 100)}%", (bbox[0], bbox[1] - 5),
                #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # out_path = os.path.join("results", filename)
                # cv2.imwrite(out_path, image)