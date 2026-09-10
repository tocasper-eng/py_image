# SKILL.md - Material Image Search System

## Skills Used
- CLIP (ViT-B/32) for image/text embedding and semantic search
- Flask for web server and API
- Gunicorn as production WSGI server
- Web Speech API for browser-based voice input (zh-TW)
- HTML5 Camera Capture for mobile photo input
- Cosine similarity for vector comparison
- Docker for containerized deployment
- Zeabur CLI for cloud deployment

## Key Patterns
- Precomputed embeddings cached in pickle for fast search
- Single-page app with no build tools required
- All frontend code inline in one HTML file
- CLIP handles both text-to-image and image-to-image search
- Normalized embedding vectors enable dot product as cosine similarity
- CPU-only PyTorch via `--extra-index-url` to reduce Docker image size
- PORT environment variable for cloud platform compatibility

## Deployment
- GitHub: https://github.com/tocasper-eng/py_image
- Zeabur: https://py-image-clip.zeabur.app/
- Zeabur Project ID: 6aa208226c3d9581b7159e85
- Zeabur Service ID: 6aa20c336c3d9581b715a12a
- Zeabur Environment ID: 6aa20822da9bc245fbcb5b87
