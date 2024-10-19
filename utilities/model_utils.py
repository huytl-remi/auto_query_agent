# utilities/model_utils.py
import torch
import clip
import faiss
import json
import os
import numpy as np
import streamlit as st
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from typing import List, Tuple, Dict, Any

from config import Config

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and FAISS index with caching
@st.cache_resource
def load_model():
    return clip.load("ViT-B/16", device=device)

@st.cache_resource
def load_faiss_index():
    return faiss.read_index("/content/drive/MyDrive/HCMC_AI/data/faiss_clip_16_full_v3.bin")

@st.cache_data
def load_id2img_fps():
    with open("/content/drive/MyDrive/HCMC_AI/data/id2img_fps_mid_full.json", "r") as f:
        return json.load(f)

# CLIP search
def encode_text(model: Any, text_query: str) -> torch.Tensor:
    text = clip.tokenize([text_query]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text)
    return text_features / text_features.norm(dim=-1, keepdim=True)

def search_image_by_text(model: Any, index: Any, text_query: str, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
    text_features = encode_text(model, text_query).cpu().numpy()
    D, I = index.search(text_features, top_k)
    return I, D

# Initialize Pinecone and OpenAI
pinecone_api_key = Config.PINECONE_API_KEY
openai_api_key = Config.OPENAI_API_KEY

# Ensure API keys are available
if not pinecone_api_key or not openai_api_key:
    raise ValueError("Pinecone and OpenAI API keys must be set in the environment variables.")

# Initialize Pinecone
pc = Pinecone(api_key=pinecone_api_key)

# Initialize OpenAI client
client = OpenAI(api_key=openai_api_key)

# Set up Pinecone index
index_name = "hcmaic-chungket"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,  # Adjust this to match your embedding dimension
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-west-2'  # Adjust region as needed
        )
    )

captioning_index = pc.Index(index_name)

def get_captioning_embedding(text_query: str) -> List[float]:
    response = client.embeddings.create(input=[text_query], model="text-embedding-3-small")
    return response.data[0].embedding

def search_image_by_text_with_captioning(text_query: str, top_k: int) -> List[str]:
    query_embedding = get_captioning_embedding(text_query)
    result = captioning_index.query(top_k=top_k, vector=query_embedding, include_metadata=True)
    data_path = "/content/drive/MyDrive/HCMC_AI/data/Mid_Frames"
    return [os.path.join(data_path, match['id']) for match in result['matches'] if os.path.exists(os.path.join(data_path, match['id']))]

# OCR search
def search_images_by_ocr(text_query: str, top_k: int) -> List[str]:
    matching_paths = []
    ocr_folder = "/content/drive/MyDrive/HCMC_AI/data/ocr"
    for root, _, files in os.walk(ocr_folder):
        for file in files:
            if file.endswith(".json"):
                try:
                    with open(os.path.join(root, file), "r") as f:
                        ocr_data = json.load(f)
                    for frame_number, ocr_texts in ocr_data.items():
                        if any(text_query.lower() in text.lower() for text in ocr_texts):
                            data_part = root.split('/')[-1]
                            video_id = file.split('.')[0]
                            image_path = f"/content/drive/MyDrive/HCMC_AI/data/Mid_Frames/{data_part}/{video_id}/{frame_number}.jpg"
                            if os.path.exists(image_path):
                                matching_paths.append(image_path)
                            if len(matching_paths) >= top_k:
                                return matching_paths
                except json.JSONDecodeError:
                    st.warning(f"Error decoding JSON file: {os.path.join(root, file)}")
                except Exception as e:
                    st.error(f"Error processing file {os.path.join(root, file)}: {str(e)}")
    return matching_paths[:top_k]

def get_image_paths(image_indices: np.ndarray, id2img_fps: Dict[str, Dict[str, str]]) -> List[str]:
    image_paths = []
    for idx in image_indices[0]:
        image_info = id2img_fps.get(str(idx))
        if image_info:
            image_paths.append(image_info.get("image_path", ""))
        else:
            st.warning(f"No image info found for index {idx}")
    return [path for path in image_paths if path]  # Filter out empty paths
