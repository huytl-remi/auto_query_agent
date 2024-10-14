# utilities/model_utils.py
import torch
import clip
import faiss
import json
import os
import streamlit as st
from openai import OpenAI
import pinecone

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
def encode_text(model, text_query):
    text = clip.tokenize([text_query]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text)
    return text_features / text_features.norm(dim=-1, keepdim=True)

def search_image_by_text(model, index, text_query, top_k):
    text_features = encode_text(model, text_query).cpu().numpy()
    D, I = index.search(text_features, top_k)
    return I, D

# Initialize Pinecone and OpenAI using API keys from Config
pinecone_api_key = Config.PINECONE_API_KEY
openai_api_key = Config.OPENAI_API_KEY

# Ensure API keys are available
if not pinecone_api_key or not openai_api_key:
    raise ValueError("Pinecone and OpenAI API keys must be set in the environment variables.")

pinecone.init(api_key=pinecone_api_key)
client = OpenAI(api_key=openai_api_key)

# Continue with Pinecone and OpenAI setup
index_name = "hcmaic-sokhao-2"
captioning_index = pinecone.Index(index_name)

def get_captioning_embedding(text_query):
    return client.embeddings.create(input=[text_query], model="text-embedding-ada-002").data[0].embedding

def search_image_by_text_with_captioning(text_query, top_k):
    query_embedding = get_captioning_embedding(text_query)
    result = captioning_index.query(top_k=top_k, vector=query_embedding, include_metadata=True)
    data_path = "/content/drive/MyDrive/HCMC_AI/data/Mid_Frames"
    return [os.path.join(data_path, match['id']) for match in result['matches'] if os.path.exists(os.path.join(data_path, match['id']))]

# OCR search
def search_images_by_ocr(text_query, top_k):
    matching_paths = []
    ocr_folder = "/content/drive/MyDrive/HCMC_AI/data/ocr"
    for root, _, files in os.walk(ocr_folder):
        for file in files:
            if file.endswith(".json"):
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
    return matching_paths[:top_k]

def get_image_paths(image_indices, id2img_fps):
    image_paths = []
    for idx in image_indices[0]:
        image_info = id2img_fps[str(idx)]
        image_paths.append(image_info["image_path"])
    return image_paths
