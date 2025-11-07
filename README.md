# Image Segmentation with Text Prompts

A full-stack application for segmenting objects in images based on natural language descriptions, using **GroundingDINO** for object detection and **SAM2** (Segment Anything Model 2) for segmentation.

## Architecture

This application consists of 5 microservices orchestrated by Docker Compose:

```
┌─────────────┐      ┌─────────────┐      ┌──────────────────┐      ┌──────────────────┐
│  Streamlit  │─────▶│   FastAPI   │─────▶│  Redis Queue 1   │─────▶│  GroundingDINO   │
│  Frontend   │      │   Backend   │      │                  │      │    Service       │
└─────────────┘      └─────────────┘      └──────────────────┘      └──────────────────┘
                            │                                                 
                            │                                                 
                            │              ┌──────────────────┐      ┌──────────────────┐
                            └─────────────▶│  Redis Queue 2   │─────▶│   SAM2 Service   │
                                           └──────────────────┘      └──────────────────┘
```

### Components

1. **Frontend (Streamlit)**: User-friendly web interface for uploading images and entering text prompts
2. **Backend (FastAPI)**: Orchestration layer that manages the two-stage processing pipeline
3. **Redis**: Message queue system with two queues:
   - Queue 1: Tasks for GroundingDINO
   - Queue 2: Tasks for SAM2
4. **GroundingDINO Service**: Detects objects in images based on text prompts and returns bounding boxes
5. **SAM2 Service**: Segments objects within the provided bounding boxes

## Model Serving Architecture

### Microservice-Based ML Serving

This application employs a **microservice architecture** for serving machine learning models, with each model running in an isolated container. This design provides several advantages over traditional monolithic ML serving:

#### Architecture Pattern: Queue-Based Worker Services

```
Request Flow:
1. User uploads image via Streamlit → FastAPI receives request
2. FastAPI encodes image (Base64) + creates task ID
3. Task pushed to Redis Queue 1 (grounding_dino_queue)
4. GroundingDINO worker pulls task, processes, returns bounding boxes
5. FastAPI receives boxes → creates new task for SAM2
6. Task pushed to Redis Queue 2 (sam2_queue)
7. SAM2 worker pulls task, segments objects, returns masks
8. FastAPI aggregates results → returns to frontend
```

#### Model Service Implementation

Each model service (`grounding_dino_service` and `sam2_service`) follows the same pattern:

1. **Model Loading**: Models are loaded once at container startup
   - GroundingDINO: `groundingdino_swint_ogc.pth` (~440MB)
   - SAM2: `sam2.1_hiera_large.pt` (~856MB)
   - Models remain in memory for fast inference

2. **Worker Loop**: Continuous polling of Redis queue
   ```python
   while True:
       task = redis.blpop('queue_name', timeout=1)  # Blocking pop
       if task:
           result = model.predict(task['image'])
           redis.rpush(f'result:{task_id}', result)
   ```

3. **Asynchronous Processing**: Non-blocking communication
   - Backend doesn't wait synchronously for model inference
   - Can handle multiple requests concurrently
   - Results retrieved via task ID when ready

#### Benefits of This Architecture

| Aspect | Benefit |
|--------|---------|
| **Isolation** | Each model runs independently; failures don't cascade |
| **Scalability** | Scale each model service independently based on demand |
| **Resource Management** | Allocate different GPU/CPU resources per service |
| **Load Balancing** | Redis queues naturally distribute work across instances |
| **Fault Tolerance** | Failed tasks can be retried; services can restart independently |
| **Monitoring** | Track queue lengths, processing times per service |

#### Deployment Models

**Single Instance (Default)**
```yaml
services:
  grounding_dino: 1 instance
  sam2: 1 instance
```
- Simple deployment
- Sequential processing
- Best for low-traffic scenarios

**Scaled Deployment**
```bash
docker-compose up --scale grounding_dino=3 --scale sam2=2
```
- Multiple workers per model
- Parallel processing of queued tasks
- Higher throughput for production use

### Model Optimization Strategies

**GPU Allocation**
- Both services configured for NVIDIA GPU access
- CUDA 11.8 runtime environment
- Automatic fallback to CPU if GPU unavailable

**Memory Management**
- Models loaded once per container (not per request)
- Shared model weights across requests
- Image data passed via Redis (encoded as Base64)

**Network Efficiency**
- All services communicate via internal Docker network
- No external network calls during inference
- Results cached in Redis with TTL (5 minutes)

## Features

- **Text-based object detection**: Describe objects in natural language
- **High-quality segmentation**: Precise object boundaries using SAM2
- **Scalable architecture**: Independent model services with Redis queues
- **Real-time processing**: Asynchronous task processing
- **Easy deployment**: Everything containerized with Docker

## Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- NVIDIA GPU with CUDA support (recommended for model inference)
- NVIDIA Container Toolkit (for GPU support in Docker)

## Installation

### 1. Clone the repository

```bash
cd project
```

### 2. Verify directory structure

```
project/
├── frontend/
│   ├── app.py
│   ├── requirements.txt
│   └── Dockerfile
├── backend/
│   ├── main.py
│   ├── requirements.txt
│   └── Dockerfile
├── grounding_dino_service/
│   ├── service.py
│   └── Dockerfile
├── sam2_service/
│   ├── service.py
│   └── Dockerfile
├── docker-compose.yml
└── README.md
```

### 3. Build and run with Docker Compose

#### With GPU support (recommended):

```bash
docker-compose up --build
```

#### Without GPU (CPU-only mode):

Use the provided CPU override file:

```bash
docker-compose -f docker-compose.no-gpu.yml up -d
```

**Note**: First build will take 15-30 minutes as it downloads models and installs dependencies.

## Usage

### 1. Access the application

Once all services are running, open your browser and navigate to:

```
http://localhost:8501
```

### 2. Segment an image

1. **Upload an image**: Click "Browse files" and select an image (JPG, PNG)
2. **Enter a text prompt**: Describe the objects you want to segment
   - Examples:
     - `"cat"`
     - `"person with red shirt"`
     - `"cat . dog . bird ."` (multiple objects)
3. **Adjust settings** (optional): Use the sidebar to tune detection thresholds
4. **Click "Segment Image"**: Wait for processing (10-30 seconds)
5. **View results**: See detected objects, confidence scores, and segmentation visualization
6. **Download**: Save the segmented image

### Example Text Prompts

- **Simple objects**: `cat`, `dog`, `person`, `car`, `tree`
- **Multiple objects**: `cat . dog . bird .` (separate with periods)
- **Descriptive**: `person with red shirt`, `white flowers`, `green apple`
- **Positional**: `leftmost person`, `front-most car`

## API Documentation

### Backend API

The FastAPI backend is available at `http://localhost:8000`

#### Endpoints

- **POST /segment**: Submit an image for segmentation
  - Parameters:
    - `image` (file): Image file
    - `text_prompt` (string): Text description of objects to segment
    - `box_threshold` (float, optional): Detection confidence threshold (default: 0.35)
    - `text_threshold` (float, optional): Text matching threshold (default: 0.25)
  - Returns: JSON with detected objects, bounding boxes, scores, and visualization

- **GET /health**: Check service health
- **GET /queue-status**: Get current Redis queue lengths
- **GET /**: API information

Interactive API documentation: `http://localhost:8000/docs`

## Configuration

### Environment Variables

Services can be configured via environment variables in `docker-compose.yml`:

- `REDIS_HOST`: Redis server hostname (default: `redis`)
- `REDIS_PORT`: Redis server port (default: `6379`)

### Model Configuration

To use different model checkpoints:

1. **GroundingDINO**: Edit `grounding_dino_service/Dockerfile` to download different weights
2. **SAM2**: Edit `sam2_service/Dockerfile` to use different SAM2 model variants (tiny, small, base, large)

## Monitoring

### Check service status

```bash
# View logs
docker-compose logs -f

# Check specific service
docker-compose logs -f frontend
docker-compose logs -f backend
docker-compose logs -f grounding_dino
docker-compose logs -f sam2

# Check Redis queue status
docker-compose exec redis redis-cli
> LLEN grounding_dino_queue
> LLEN sam2_queue
```

### Health checks

- Backend health: `http://localhost:8000/health`
- Queue status: `http://localhost:8000/queue-status`

## Troubleshooting

### Common Issues

1. **GPU not detected**
   - Install NVIDIA Container Toolkit
   - Verify GPU: `nvidia-smi`
   - Check Docker GPU access: `docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi`

2. **Out of memory**
   - Reduce image size before uploading
   - Use smaller model variants
   - Switch to CPU mode (slower but uses less memory)

3. **Service timeout**
   - Check if model services are running: `docker-compose ps`
   - View logs for errors: `docker-compose logs grounding_dino sam2`
   - Increase timeout in `backend/main.py` if needed

4. **Build failures**
   - Ensure stable internet connection (models are downloaded during build)
   - Clear Docker cache: `docker-compose down -v && docker system prune -a`
   - Retry build: `docker-compose up --build --force-recreate`

## Development

### Running services locally (without Docker)

Each service can be run independently for development:

```bash
# Terminal 1: Redis
docker run -p 6379:6379 redis:7-alpine

# Terminal 2: Backend
cd backend
pip install -r requirements.txt
uvicorn main:app --reload

# Terminal 3: Frontend
cd frontend
pip install -r requirements.txt
streamlit run app.py

# Note: Model services require their full environments (see Dockerfiles)
```

## Performance

### Inference Time Breakdown

Inference time varies significantly based on hardware acceleration and image characteristics.

#### GPU Inference (NVIDIA CUDA)

| Component | Time (seconds) | Notes |
|-----------|---------------|-------|
| **GroundingDINO** | 1-3s | Object detection, scales with image resolution |
| **SAM2** | 2-7s | Segmentation, scales with number of objects |
| **Network/Queue** | <1s | Redis communication overhead |
| **Total Pipeline** | **3-10s** | End-to-end per image |

**Recommended GPU**: NVIDIA RTX 3060 or better (8GB+ VRAM)

#### CPU-Only Inference

| Component | Time (seconds) | Notes |
|-----------|---------------|-------|
| **GroundingDINO** | 10-20s | Slower due to lack of hardware acceleration |
| **SAM2** | 15-40s | Most computationally intensive component |
| **Network/Queue** | <1s | Minimal overhead |
| **Total Pipeline** | **25-60s** | End-to-end per image |

**Note**: CPU inference is 5-8x slower than GPU inference.

#### Factors Affecting Performance

1. **Image Resolution**
   - 512×512: GPU ~3-5s, CPU ~25-30s
   - 1024×1024: GPU ~5-8s, CPU ~40-50s
   - 2048×2048+: GPU ~8-10s, CPU ~60-90s

2. **Number of Objects Detected**
   - 1-2 objects: Minimal impact
   - 5+ objects: SAM2 processing increases linearly
   - Each additional object: +0.5-1s (GPU), +3-5s (CPU)

3. **Text Prompt Complexity**
   - Simple prompts ("cat"): Faster detection
   - Complex prompts ("person with red shirt"): Slightly slower
   - Multiple objects ("cat . dog . bird ."): Increases detection time

4. **Hardware Specifications**

   **GPU Performance Tiers**:
   - Entry (GTX 1660, RTX 3050): 8-12s per image
   - Mid-range (RTX 3060, RTX 4060): 5-8s per image
   - High-end (RTX 4080, A100): 3-5s per image

   **CPU Performance Tiers**:
   - Older/Low-core (4 cores): 60-90s per image
   - Mid-range (8-16 cores, Ryzen 5/i5): 40-50s per image
   - High-end (16+ cores, Ryzen 9/i9): 25-35s per image

### Throughput Estimates

#### Single Instance Deployment

| Hardware | Images/Minute | Images/Hour | Optimal Use Case |
|----------|---------------|-------------|------------------|
| High-end GPU | 6-10 | 360-600 | Production, real-time processing |
| Mid-range GPU | 4-6 | 240-360 | Development, moderate traffic |
| CPU-only | 1-2 | 60-120 | Testing, low-traffic scenarios |

#### Scaled Deployment (3× workers each)

| Hardware | Images/Minute | Images/Hour | Concurrent Capacity |
|----------|---------------|-------------|---------------------|
| High-end GPU | 18-30 | 1080-1800 | High-traffic production |
| Mid-range GPU | 12-18 | 720-1080 | Medium-traffic production |
| CPU-only | 3-6 | 180-360 | Batch processing |

### Performance Optimization Tips

#### For CPU Deployments

If you must use CPU-only inference, consider these optimizations:

1. **Use Smaller Model Variant**

   Edit `sam2_service/Dockerfile`:
   ```dockerfile
   # Replace large model with small variant
   RUN wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt
   ```

   Edit `sam2_service/service.py`:
   ```python
   MODEL_CHECKPOINT = "/workspace/sam2/checkpoints/sam2.1_hiera_small.pt"
   MODEL_CONFIG = "/workspace/sam2/configs/sam2.1/sam2.1_hiera_s.yaml"
   ```

   **Impact**: Reduces SAM2 time by 40-50% with minimal accuracy loss

2. **Increase Backend Timeout**

   Edit `backend/main.py`:
   ```python
   timeout = 120  # Increase from 60 to 120 seconds
   ```

3. **Resize Images**

   Add image preprocessing to limit max dimension to 1024px

4. **Enable Multi-threading**

   Set environment variables in `docker-compose.yml`:
   ```yaml
   environment:
     - OMP_NUM_THREADS=4
     - MKL_NUM_THREADS=4
   ```

#### For GPU Deployments

1. **Batch Processing**: Process multiple images concurrently
2. **Mixed Precision**: Enable FP16 inference (requires code changes)
3. **Model Quantization**: Use INT8 quantized models (advanced)

### System Requirements

| Configuration | CPU | RAM | GPU VRAM | Storage |
|--------------|-----|-----|----------|---------|
| **Minimum (CPU)** | 4 cores | 8GB | - | 10GB |
| **Recommended (GPU)** | 8 cores | 16GB | 8GB | 15GB |
| **Production (GPU)** | 16+ cores | 32GB | 16GB+ | 20GB |

### Monitoring Performance

Check processing times in real-time:

```bash
# Watch service logs for timing information
docker-compose logs -f grounding_dino | grep "completed"
docker-compose logs -f sam2 | grep "completed"

# Monitor queue lengths
curl http://localhost:8000/queue-status
```

**Performance Metrics Added** (in service logs):
- Task received timestamp
- Processing duration
- Objects detected/segmented count

## Scaling

To run multiple instances of model services:

```bash
docker-compose up --scale grounding_dino=2 --scale sam2=2
```

## References

- **GroundingDINO**: [GitHub](https://github.com/IDEA-Research/GroundingDINO) | [Paper](https://arxiv.org/pdf/2303.05499)
- **SAM2**: [GitHub](https://github.com/facebookresearch/sam2) | [Paper](https://arxiv.org/abs/2408.00714)

## License

This project uses models from Meta (SAM2) and IDEA Research (GroundingDINO). Please refer to their respective licenses.

## Support

For issues and questions:
- Check troubleshooting section
- Review logs: `docker-compose logs`
- Open an issue with logs and error messages
