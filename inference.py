import os
import uuid
import torch
import chromadb
from collections import deque
from PIL import Image
from unsloth import FastVisionModel
from model_loading import GlobalModelLoader
from classifier import classify_image
from query_classifier import classify_query
from vqa_inference import run_vqa_inference
from geo_ground import run_grounding
from fcc import run_fcc_pipeline
# Assume necessary imports for your specific geo/yolo models
# from ultralytics import YOLO 
# Set this environment variable BEFORE running your main Python script (e.g., in your shell or Colab notebook cell)
# ==========================================
# 1. MEMORY SYSTEMS (Adapted from your file)
# ==========================================

class SessionMemoryManager:
    """
    Manages memory isolation. 
    New Image = New Session ID = New Chroma Collection + New Short-Term Buffer.
    """
    def __init__(self, persist_dir="./data/chroma_db"):
        self.chroma_client = chromadb.PersistentClient(path=persist_dir)
        self.embedder = None # Load your embedding model here if needed
        # We maintain a dictionary of active sessions or just switch context
        self.current_session_id = None
        self.long_term_collection = None
        self.short_term_buffer = deque(maxlen=5)

    def start_new_session(self, image_id):
        """Called when a new image is detected."""
        print(f"ðŸ”„ New Image detected. Switching Session to: {image_id}")
        self.current_session_id = image_id
        
        # 1. Reset Short Term Memory
        self.short_term_buffer.clear()
        
        # 2. Get/Create Isolated Long Term Collection for this specific image
        # Sanitizing id for chroma (alphanumeric only)
        safe_id = "session_" + "".join(c for c in image_id if c.isalnum())
        self.long_term_collection = self.chroma_client.get_or_create_collection(name=safe_id)

    def add_turn(self, user_text, bot_text):
        if not self.long_term_collection:
            return
            
        # Add to Short Term (RAM)
        blob = f"User: {user_text}\nBot: {bot_text}"
        self.short_term_buffer.append(blob)
        
        # Add to Long Term (Disk)
        # Note: You need a real embedding function here usually
        # For prototype, we assume chroma's default or the one from your file
        unique_id = str(uuid.uuid4())
        self.long_term_collection.add(
            documents=[blob],
            ids=[unique_id],
            metadatas=[{"timestamp": "now"}] # Add real timestamp
        )

    def get_context(self, query):
        """Combines short-term buffer + relevant long-term retrieval"""
        # 1. Short Term
        context_str = "\n".join(self.short_term_buffer)
        
        # 2. Long Term (RAG)
        if self.long_term_collection:
            results = self.long_term_collection.query(
                query_texts=[query],
                n_results=2
            )
            if results['documents']:
                rag_context = "\n".join(results['documents'][0])
                context_str = f"Old Memories:\n{rag_context}\n\nRecent Chat:\n{context_str}"
        
        return context_str

# ==========================================
# 2. MODEL LOADERS & CLASSIFIERS
# ==========================================

from model_loading import GlobalModelLoader # Import the new file we just made

class ModelZoo:
    def __init__(self):
        self.loader = GlobalModelLoader()
        self.loaded = False

    def load_models(self):
        print("ðŸ—ï¸ Initializing Model Zoo...")
        
        # Define your paths here (Update these with your real paths from Colab/Drive)
        config = {
            'classifier': '/content/tati/temp/classifiers/final_model_all_classes.keras',
            'fcc': '/content/tati/temp/fcc model/fcc_best_model.pt',
            'qwen': 'unsloth/Qwen3-VL-32B-Instruct-unsloth-bnb-4bit', # or your specific base
            'lora_sar': '/content/drive/MyDrive/SAR_LORA_ADAPTER',
            'lora_optical': '/content/drive/MyDrive/Qwen_for_optical',
            'lora_fcc': '/content/drive/MyDrive/Qwen_for_optical',

            'yolo_base': '/content/tati/Final_Grounding/yolo11n-obb.pt',  # Path to base YOLO model
            'yolo_lora': '/content/tati/Final_Grounding/lora_weights.pt',  # Path to YOLO LoRA weights
            'qwen_grounding': 'Qwen/Qwen2-VL-7B-Instruct'  # Qwen model for refinement
            # Add others as needed
        }
        
        self.loader.load_all(config)
        self.loaded = True
        return self
        
    # Helper properties to access models easily
    @property
    def classifier(self): return self.loader.classifier
    
    @property
    def qwen_model(self): return self.loader.qwen_model
    
    @property
    def qwen_tokenizer(self): return self.loader.qwen_tokenizer
    
    @property
    def geoground(self): return self.loader.geoground_pipeline

    @property
    def fcc_net(self): return self.loader.fcc_model

# ==========================================
# 4. MAIN INFERENCE PIPELINE
# ==========================================
class InferencePipeline:
    def __init__(self):
        self.zoo = ModelZoo()
        self.memory_manager = SessionMemoryManager()
        self.previous_image_path = None

    def initialize(self):
        if not self.zoo.loaded:
            self.zoo.load_models()

    def run(self, image_path, user_query):
        self.initialize()
        responses = []

        # --- MEMORY & SESSION HANDLING ---
        if image_path != self.previous_image_path:
            image_id = os.path.basename(image_path)
            self.memory_manager.start_new_session(image_id)
            self.previous_image_path = image_path
        
        # Retrieve Memory Context
        memory_context = self.memory_manager.get_context(user_query) + f"\nUser Question: {user_query}"

        # --- CLASSIFICATION ---
        # 1. Determine Image Type (SAR, Optical, FCC)
        img_type = classify_image(self.zoo, image_path)
        
        # 2. Determine User Intent (VQA, Grounding)
        tasks = classify_query(self.zoo, user_query)


        
        print(f"ðŸš€ Routing: Image=[{img_type}] | Tasks={tasks}")          

        # --- ROUTING LOGIC ---
        for task in tasks:
            task_type = task['category'][0] 
            sub_query = task['query']
            
            # --- CASE A: CAPTIONING / VQA ---
            if img_type == "fcc" and task_type != "GROUNDING" :
              print("FCC")
              # Call the imported function
              answer = run_fcc_pipeline(self.zoo, image_path, sub_query)
              responses.append(f"[{task_type}] {answer}")
            elif task_type == "CAPTIONING":
              adapter_key = img_type 
              # sub_query = add_vqa_instructions(sub_query)
              print(sub_query)
              # Call the imported function
              answer = run_vqa_inference(self.zoo, adapter_key, image_path, sub_query,memory_context)
              responses.append(f"[{task_type}] {answer}")
            elif task_type == "VQA/Semantic":
              adapter_key = img_type 
              sub_query = sub_query + " Provide a concise answer of exactly 4-5 words. Be specific and descriptive; do not include extra explanation or punctuation."
              print(sub_query)
              # Call the imported function
              answer = run_vqa_inference(self.zoo, adapter_key, image_path, sub_query,memory_context)
              responses.append(f"[{task_type}] {answer}")
            elif task_type == "VQA/Binary":
              adapter_key = img_type 
              sub_query = sub_query + " Answer only with exactly 'Yes' or 'No'. Do not add explanations or other words."
              print(sub_query)
              # Call the imported function
              answer = run_vqa_inference(self.zoo, adapter_key, image_path, sub_query, memory_context)
              responses.append(f"[{task_type}] {answer}")
            elif task_type == "VQA/Numeric":
              adapter_key = img_type 
              sub_query = sub_query + " Answer using digits only. Provide the exact numeric value and nothing else â€” no words, units, or explanations."
              print(sub_query)
              # Call the imported function
              answer = run_vqa_inference(self.zoo, adapter_key, image_path, sub_query, memory_context)
              responses.append(f"[{task_type}] {answer}")
            # --- CASE B: GROUNDING ---
            elif task_type == "GROUNDING":
                # Call the imported function
                grounding_result = run_grounding(self.zoo, image_path, sub_query)
                responses.append(f"[GROUNDING] {grounding_result}")

        # --- FINALIZATION ---
        final_response = "\n".join(responses)
        self.memory_manager.add_turn(user_query, final_response)
        
        return final_response

# ==========================================
# 5. EXECUTION EXAMPLE
# ==========================================

if __name__ == "__main__":
    pipeline = InferencePipeline()
    
    # 1. First Conversation (Image A)
    print("--- User sends Image A ---")
    img_a = "3f4936ab-8c66-4df8-9c7f-dcfd828806a8.png"
    res1 = pipeline.run(img_a, "Locate the tennis courts in the image")
    print(f"Bot: {res1}\n")
    
    res2 = pipeline.run(img_a, "explain the network") # Should use memory of "city"
    print(f"Bot: {res2}\n")

    # # 2. Second Conversation (Image B) - MEMORY SHOULD RESET
    # print("--- User sends Image B (New Context) ---")
    # img_b = "/content/images_50/Images_train/04661_0000.png"
    # res3 = pipeline.run(img_b, "What is the no. of tennis court?")
    # print(f"Bot: {res3}\n")
