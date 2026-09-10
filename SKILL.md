# SKILL.md - Material Image Search System

## Skills Used
- CLIP (ViT-B/32) for image/text embedding and semantic search
- Flask for web server and API
- Web Speech API for browser-based voice input (zh-TW)
- HTML5 Camera Capture for mobile photo input
- Cosine similarity for vector comparison

## Key Patterns
- Precomputed embeddings cached in pickle for fast search
- Single-page app with no build tools required
- All frontend code inline in one HTML file
- CLIP handles both text-to-image and image-to-image search
- Normalized embedding vectors enable dot product as cosine similarity
