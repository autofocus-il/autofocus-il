from ultralytics import YOLO

model = YOLO("/path/to/project_root/saliency_pipeline/models/yolo_widowx_gripper.pt")

# source can be a path of directory or a single image file
results = model.predict(source="/path/to/images")

CLASS_MAP = {0: "gripper"}
CONF_THRESHOLD = 0.85

BDV2_IMAGE_HEIGHT = 480
BDV2_IMAGE_WIDTH = 640

def xyxy_to_normalized_center(xyxy, image_height, image_width):
    x1, y1, x2, y2 = xyxy
    center_x = (x1 + x2) / 2 / image_width
    center_y = (y1 + y2) / 2 / image_height
    return [center_x, center_y]

for result in results:
    if result.boxes is not None and len(result.boxes) > 0:
        for i in range(len(result.boxes)):
            # bdv2: "image_height": 480, "image_width": 640,
            conf = result.boxes.conf[i].item()
            if conf < CONF_THRESHOLD:
                continue
            
            xyxy = result.boxes.xyxy[i].tolist()
            cls_id = int(result.boxes.cls[i].item())
            
            center_coords = xyxy_to_normalized_center(xyxy, BDV2_IMAGE_HEIGHT, BDV2_IMAGE_WIDTH)
            print(f"Center coords: {center_coords}, Confidence: {conf:.3f}, Class: {CLASS_MAP[cls_id]}")
            # output: Center coords: [0.3881789207458496, 0.7266825993855794], Confidence: 0.864, Class: gripper