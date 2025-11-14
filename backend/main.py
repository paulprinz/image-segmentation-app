import os
import json
import uuid
import base64
import time
import io
import redis
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="Image Segmentation API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Redis connection
REDIS_HOST = os.getenv('REDIS_HOST', 'redis')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=False)

class SegmentationRequest(BaseModel):
    text_prompt: str
    box_threshold: Optional[float] = 0.35
    text_threshold: Optional[float] = 0.25

@app.get("/")
def read_root():
    return {"message": "Image Segmentation API", "status": "running"}

@app.get("/health")
def health_check():
    try:
        redis_client.ping()
        return {"status": "healthy", "redis": "connected"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

@app.post("/segment")
async def segment_image(
    image: UploadFile = File(...),
    text_prompt: str = Form(...),
    box_threshold: float = Form(0.35),
    text_threshold: float = Form(0.25)
):
    """
    Main endpoint to segment objects in an image based on text prompt.

    This endpoint orchestrates the two-stage process:
    1. GroundingDINO detects objects based on text prompt
    2. SAM2 segments the detected objects
    """
    try:
        # Generate unique task ID
        task_id = str(uuid.uuid4())

        # Read image and downsample to 256x256 to reduce GPU memory usage
        image_bytes = await image.read()
        original_image = Image.open(io.BytesIO(image_bytes))
        original_size = original_image.size  # (width, height)

        # Resize to 256x256 using high-quality Lanczos resampling
        resized_image = original_image.resize((256, 256), Image.Resampling.LANCZOS)

        # Convert back to bytes
        buffer = io.BytesIO()
        resized_image.save(buffer, format=original_image.format or 'PNG')
        resized_bytes = buffer.getvalue()
        image_b64 = base64.b64encode(resized_bytes).decode('utf-8')

        print(f"Task {task_id}: Starting segmentation with prompt '{text_prompt}'")
        print(f"Task {task_id}: Image resized from {original_size} to (256, 256)")

        # Stage 1: GroundingDINO Object Detection
        print(f"Task {task_id}: Sending to GroundingDINO...")
        grounding_task = {
            'task_id': task_id,
            'image': image_b64,
            'text_prompt': text_prompt,
            'box_threshold': box_threshold,
            'text_threshold': text_threshold
        }

        # Push to GroundingDINO queue
        redis_client.rpush('grounding_dino_queue', json.dumps(grounding_task))

        # Wait for GroundingDINO result (with timeout)
        timeout = 600  # 600 seconds timeout (10 minutes for CPU processing on Apple Silicon)
        start_time = time.time()

        while time.time() - start_time < timeout:
            result_data = redis_client.blpop(f'grounding_dino_result:{task_id}', timeout=1)
            if result_data:
                _, result_json = result_data
                grounding_result = json.loads(result_json)
                break
            time.sleep(0.1)
        else:
            raise HTTPException(status_code=504, detail="GroundingDINO processing timeout")

        if grounding_result['status'] == 'error':
            raise HTTPException(status_code=500, detail=f"GroundingDINO error: {grounding_result['error']}")

        boxes = grounding_result['result']['boxes']
        labels = grounding_result['result']['labels']
        scores = grounding_result['result']['scores']

        print(f"Task {task_id}: GroundingDINO found {len(boxes)} objects")

        # Check if any objects were detected
        if len(boxes) == 0:
            return JSONResponse({
                'task_id': task_id,
                'status': 'completed',
                'message': 'No objects detected matching the text prompt',
                'detected_objects': 0,
                'boxes': [],
                'labels': [],
                'visualization': None
            })

        # Stage 2: SAM2 Segmentation
        print(f"Task {task_id}: Sending to SAM2...")
        sam2_task = {
            'task_id': task_id,
            'image': image_b64,
            'boxes': boxes
        }

        # Push to SAM2 queue
        redis_client.rpush('sam2_queue', json.dumps(sam2_task))

        # Wait for SAM2 result (with timeout)
        start_time = time.time()

        while time.time() - start_time < timeout:
            result_data = redis_client.blpop(f'sam2_result:{task_id}', timeout=1)
            if result_data:
                _, result_json = result_data
                sam2_result = json.loads(result_json)
                break
            time.sleep(0.1)
        else:
            raise HTTPException(status_code=504, detail="SAM2 processing timeout")

        if sam2_result['status'] == 'error':
            raise HTTPException(status_code=500, detail=f"SAM2 error: {sam2_result['error']}")

        print(f"Task {task_id}: Segmentation completed successfully")

        # Prepare final response
        response = {
            'task_id': task_id,
            'status': 'completed',
            'detected_objects': len(boxes),
            'boxes': boxes,
            'labels': labels,
            'scores': scores,
            'visualization': sam2_result['result']['visualization']
        }

        return JSONResponse(response)

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in segment_image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/queue-status")
def get_queue_status():
    """
    Get the current status of Redis queues.
    """
    try:
        grounding_queue_length = redis_client.llen('grounding_dino_queue')
        sam2_queue_length = redis_client.llen('sam2_queue')

        return {
            'grounding_dino_queue': grounding_queue_length,
            'sam2_queue': sam2_queue_length
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
