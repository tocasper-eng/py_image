import os
import pickle
import logging
from pathlib import Path

import clip
import numpy as np
import torch
from flask import Flask, jsonify, render_template, request, send_from_directory
from PIL import Image

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
IMAGE_FOLDER = BASE_DIR / "image_library"
INDEX_CACHE = BASE_DIR / "index_cache"
EMBEDDINGS_FILE = INDEX_CACHE / "embeddings.pkl"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Loading CLIP model on {device}...")
model, preprocess = clip.load("ViT-B/32", device=device)
logger.info("CLIP model loaded.")

# Global index storage
image_filenames = []
image_embeddings = None


def build_index():
    """Scan image_library folder and build CLIP embeddings for all images."""
    global image_filenames, image_embeddings

    IMAGE_FOLDER.mkdir(parents=True, exist_ok=True)
    INDEX_CACHE.mkdir(parents=True, exist_ok=True)

    files = sorted([
        f.name for f in IMAGE_FOLDER.iterdir()
        if f.suffix.lower() in SUPPORTED_EXTENSIONS
    ])

    if not files:
        logger.warning("No images found in image_library/")
        image_filenames = []
        image_embeddings = None
        return 0

    logger.info(f"Building index for {len(files)} images...")
    embeddings = []
    valid_files = []

    for fname in files:
        filepath = IMAGE_FOLDER / fname
        try:
            img = preprocess(Image.open(filepath).convert("RGB")).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = model.encode_image(img)
                emb = emb / emb.norm(dim=-1, keepdim=True)
                embeddings.append(emb.cpu().numpy().flatten())
            valid_files.append(fname)
        except Exception as e:
            logger.warning(f"Skipping {fname}: {e}")

    image_filenames = valid_files
    image_embeddings = np.array(embeddings) if embeddings else None

    # Save cache
    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump({"filenames": image_filenames, "embeddings": image_embeddings}, f)

    logger.info(f"Index built: {len(image_filenames)} images indexed.")
    return len(image_filenames)


def load_index():
    """Load cached index or build new one."""
    global image_filenames, image_embeddings

    if EMBEDDINGS_FILE.exists():
        logger.info("Loading cached index...")
        with open(EMBEDDINGS_FILE, "rb") as f:
            data = pickle.load(f)
        image_filenames = data["filenames"]
        image_embeddings = data["embeddings"]
        logger.info(f"Loaded {len(image_filenames)} images from cache.")
    else:
        build_index()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/search/text", methods=["POST"])
def search_text():
    """Search images by text query using CLIP."""
    data = request.get_json()
    query = data.get("query", "").strip()

    if not query:
        return jsonify({"error": "Query is empty"}), 400

    if image_embeddings is None or len(image_filenames) == 0:
        return jsonify({"results": [], "message": "Image library is empty"})

    # Encode text query
    text_tokens = clip.tokenize([query]).to(device)
    with torch.no_grad():
        text_emb = model.encode_text(text_tokens)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
        text_emb = text_emb.cpu().numpy().flatten()

    # Compute cosine similarity
    similarities = image_embeddings @ text_emb

    # Sort by similarity descending
    sorted_indices = np.argsort(similarities)[::-1]

    results = []
    for idx in sorted_indices:
        results.append({
            "filename": image_filenames[idx],
            "similarity": round(float(similarities[idx]) * 100, 1)
        })

    return jsonify({"results": results})


@app.route("/api/search/image", methods=["POST"])
def search_image():
    """Search similar images by uploading an image."""
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No image selected"}), 400

    if image_embeddings is None or len(image_filenames) == 0:
        return jsonify({"results": [], "message": "Image library is empty"})

    try:
        img = preprocess(Image.open(file.stream).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            img_emb = model.encode_image(img)
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
            img_emb = img_emb.cpu().numpy().flatten()
    except Exception as e:
        return jsonify({"error": f"Failed to process image: {str(e)}"}), 400

    # Compute cosine similarity
    similarities = image_embeddings @ img_emb

    # Sort by similarity descending
    sorted_indices = np.argsort(similarities)[::-1]

    results = []
    for idx in sorted_indices:
        results.append({
            "filename": image_filenames[idx],
            "similarity": round(float(similarities[idx]) * 100, 1)
        })

    return jsonify({"results": results})


@app.route("/api/images/<path:filename>")
def serve_image(filename):
    """Serve an image from the image library."""
    return send_from_directory(IMAGE_FOLDER, filename)


@app.route("/api/reindex", methods=["POST"])
def reindex():
    """Rebuild the image index."""
    count = build_index()
    return jsonify({"message": "Index rebuilt successfully", "count": count})


load_index()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
