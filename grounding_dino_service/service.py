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
from torchvision.ops import box_convert

# Add GroundingDINO to path
sys.path.append('/workspace/GroundingDINO')
from groundingdino.util.inference import load_model, predict

# Redis connection
REDIS_HOST = os.getenv('REDIS_HOST', 'redis')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))

redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=False)

# Load GroundingDINO model
print("Loading GroundingDINO model...")
MODEL_CONFIG = "/workspace/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
MODEL_CHECKPOINT = "/workspace/GroundingDINO/weights/groundingdino_swint_ogc.pth"

# Use CPU to save GPU memory for SAM2 (GPU has only 4GB)
# CUDA_VISIBLE_DEVICES="" is set in Dockerfile to force CPU-only mode
device = 'cpu'
model = load_model(MODEL_CONFIG, MODEL_CHECKPOINT, device=device)
print(f"GroundingDINO model loaded successfully on {device}!", flush=True)

def process_image(image_bytes, text_prompt, box_threshold=0.35, text_threshold=0.25):
    """
    Process image with GroundingDINO to detect objects based on text prompt.

    Args:
        image_bytes: Image data in bytes
        text_prompt: Text description of objects to detect
        box_threshold: Confidence threshold for bounding boxes
        text_threshold: Confidence threshold for text matching

    Returns:
        Dictionary containing bounding boxes, scores, and labels
    """
    # Convert bytes to PIL Image
    image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_np = np.array(image_pil)

    # Prepare image for GroundingDINO
    # Convert to torch tensor and normalize, ensure it's on CPU
    transform_image = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
    transform_image = transform_image.cpu()  # Ensure CPU

    # Get image dimensions
    h, w = image_np.shape[:2]

    # Predict bounding boxes
    boxes, logits, phrases = predict(
        model=model,
        image=transform_image,
        caption=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        device='cpu'  # Explicitly pass device
    )

    # Convert boxes to xyxy format (needed for SAM2), ensure CPU tensors
    boxes_unnorm = boxes.cpu() * torch.Tensor([w, h, w, h]).cpu()
    boxes_xyxy = box_convert(boxes=boxes_unnorm, in_fmt="cxcywh", out_fmt="xyxy").cpu().numpy()

    # Prepare result
    result = {
        'boxes': boxes_xyxy.tolist(),
        'scores': logits.numpy().tolist(),
        'labels': phrases,
        'image_shape': [h, w]
    }

    return result

def main():
    print("GroundingDINO Service started. Listening for tasks...")

    while True:
        try:
            # Wait for task from Redis queue (blocking call with 1 second timeout)
            task_data = redis_client.blpop('grounding_dino_queue', timeout=1)

            if task_data is None:
                continue

            # Parse task
            _, task_json = task_data
            task = json.loads(task_json)

            task_id = task['task_id']
            image_b64 = task['image']
            text_prompt = task['text_prompt']
            box_threshold = task.get('box_threshold', 0.35)
            text_threshold = task.get('text_threshold', 0.25)

            print(f"Processing task {task_id} with prompt: '{text_prompt}'")

            # Decode image
            image_bytes = base64.b64decode(image_b64)

            # Process with GroundingDINO
            result = process_image(image_bytes, text_prompt, box_threshold, text_threshold)

            # Prepare response
            response = {
                'task_id': task_id,
                'status': 'success',
                'result': result
            }

            # Push to response queue
            redis_client.rpush(f'grounding_dino_result:{task_id}', json.dumps(response))
            redis_client.expire(f'grounding_dino_result:{task_id}', 300)  # Expire after 5 minutes

            print(f"Task {task_id} completed. Found {len(result['boxes'])} objects.")

        except Exception as e:
            print(f"Error processing task: {str(e)}")
            if 'task_id' in locals():
                error_response = {
                    'task_id': task_id,
                    'status': 'error',
                    'error': str(e)
                }
                redis_client.rpush(f'grounding_dino_result:{task_id}', json.dumps(error_response))
                redis_client.expire(f'grounding_dino_result:{task_id}', 300)

if __name__ == '__main__':
    main()
