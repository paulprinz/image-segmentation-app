# Image Segmentation with Text Prompts

A full-stack application for segmenting objects in images based on natural language descriptions, using **GroundingDINO** for object detection and **SAM2** (Segment Anything Model 2) for segmentation.

## Architecture

This application consists of 4 microservices and 2 queues orchestrated by Docker Compose:

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
### Build and run with Docker Compose

```bash
docker-compose up --build
```

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
