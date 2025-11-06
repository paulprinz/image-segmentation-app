import os
import sys
import json
import redis
import numpy as np
import torch
import cv2
from PIL import Image
import base64
import io
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Add SAM2 to path
sys.path.append('/workspace/sam2')
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Redis connection
REDIS_HOST = os.getenv('REDIS_HOST', 'redis')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))

redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=False)

# Load SAM2 model
print("Loading SAM2 model...")
MODEL_CONFIG = "/workspace/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
MODEL_CHECKPOINT = "/workspace/sam2/checkpoints/sam2.1_hiera_large.pt"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
sam2_model = build_sam2(MODEL_CONFIG, MODEL_CHECKPOINT, device=device)
predictor = SAM2ImagePredictor(sam2_model)
print(f"SAM2 model loaded successfully on {device}!")

def create_segmentation_overlay(image_np, masks, boxes):
    """
    Create a visualization of the segmentation results.

    Args:
        image_np: Original image as numpy array
        masks: Segmentation masks
        boxes: Bounding boxes

    Returns:
        Image with segmentation overlay as bytes
    """
    np.random.seed(3)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image_np)

    # Draw masks
    for mask in masks:
        if len(mask.shape) > 2:
            mask = mask.squeeze(0)

        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        h, w = mask.shape[-2:]
        mask = mask.astype(np.uint8)
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

        # Draw contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2)

        ax.imshow(mask_image)

    # Draw bounding boxes
    for box in boxes:
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        rect = patches.Rectangle((x0, y0), w, h, linewidth=2, edgecolor='green', facecolor='none')
        ax.add_patch(rect)

    ax.axis('off')
    plt.tight_layout()

    # Convert plot to bytes
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    plt.close(fig)

    return buf.getvalue()

def process_segmentation(image_bytes, boxes):
    """
    Process image with SAM2 to segment objects in bounding boxes.

    Args:
        image_bytes: Image data in bytes
        boxes: List of bounding boxes in xyxy format

    Returns:
        Dictionary containing segmentation masks and visualization
    """
    # Convert bytes to PIL Image then to numpy
    image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_np = np.array(image_pil)

    # Set image for SAM2
    predictor.set_image(image_np)

    # Convert boxes to numpy array
    boxes_np = np.array(boxes)

    # Predict masks
    masks, scores, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=boxes_np,
        multimask_output=False,
    )

    # Create visualization
    viz_bytes = create_segmentation_overlay(image_np, masks, boxes_np)

    # Convert masks to list (for JSON serialization)
    masks_list = [mask.squeeze(0).astype(np.uint8).tolist() for mask in masks]

    result = {
        'masks': masks_list,
        'scores': scores.tolist(),
        'visualization': base64.b64encode(viz_bytes).decode('utf-8')
    }

    return result

def main():
    print("SAM2 Service started. Listening for tasks...")

    while True:
        try:
            # Wait for task from Redis queue (blocking call with 1 second timeout)
            task_data = redis_client.blpop('sam2_queue', timeout=1)

            if task_data is None:
                continue

            # Parse task
            _, task_json = task_data
            task = json.loads(task_json)

            task_id = task['task_id']
            image_b64 = task['image']
            boxes = task['boxes']

            print(f"Processing task {task_id} with {len(boxes)} bounding boxes")

            # Decode image
            image_bytes = base64.b64decode(image_b64)

            # Process with SAM2
            result = process_segmentation(image_bytes, boxes)

            # Prepare response
            response = {
                'task_id': task_id,
                'status': 'success',
                'result': result
            }

            # Push to response queue
            redis_client.rpush(f'sam2_result:{task_id}', json.dumps(response))
            redis_client.expire(f'sam2_result:{task_id}', 300)  # Expire after 5 minutes

            print(f"Task {task_id} completed. Generated {len(boxes)} masks.")

        except Exception as e:
            print(f"Error processing task: {str(e)}")
            if 'task_id' in locals():
                error_response = {
                    'task_id': task_id,
                    'status': 'error',
                    'error': str(e)
                }
                redis_client.rpush(f'sam2_result:{task_id}', json.dumps(error_response))
                redis_client.expire(f'sam2_result:{task_id}', 300)

if __name__ == '__main__':
    main()
