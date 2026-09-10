# Material Image Search System

## Overview
A Google-style image search system using CLIP (ViT-B/32) for semantic similarity search.
Supports text search, voice input, and camera/photo upload.

## Quick Start
```bash
pip install -r requirements.txt
python app.py
# Open http://localhost:5000
```

## Project Structure
- `app.py` - Flask backend with CLIP model and API endpoints
- `templates/index.html` - Frontend UI (HTML/CSS/JS inline)
- `image_library/` - Put your images here (jpg, png, bmp, webp)
- `index_cache/` - Auto-generated embedding cache
- `requirements.txt` - Python dependencies

## API Endpoints
- `POST /api/search/text` - Text-to-image search (`{"query": "..."}`)
- `POST /api/search/image` - Image-to-image search (multipart form-data, field: `image`)
- `GET /api/images/<filename>` - Serve image file from library
- `POST /api/reindex` - Rebuild CLIP embedding index

## Configuration
- `IMAGE_FOLDER`: Change in `app.py` to point to your image directory (default: `./image_library`)
- Default port: 5000
- CLIP model: ViT-B/32 (auto-downloaded on first run, ~600MB)

## Adding Images
1. Put image files into `image_library/`
2. Click "Rebuild Index" button on the web page, or POST to `/api/reindex`

## Tech Stack
Python, Flask, PyTorch, CLIP (ViT-B/32), Web Speech API, Pillow, NumPy
