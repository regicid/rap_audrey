import os
import pandas as pd
import json
from typing import List, Literal
from pydantic import BaseModel

# 1. Environment and vLLM Imports
os.environ["VLLM_USE_V1"] = "1"
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

# --- 1. DEFINE OUTPUT SCHEMA ---
class InequalityMention(BaseModel):
    lines: str
    type: Literal["inégalités des chances", "inégalités raciales", "inégalités de genre", "autre"]
    stance: Literal["reproduction", "dénonciation", "constat"]
    justification: str
    confidence: Literal["faible", "moyenne", "élevée"]

class InequalityAnalysis(BaseModel):
    mentions: List[InequalityMention]

# --- 2. MAIN FUNCTION ---
def main():
    # Load dataset
    try:
        print("Loading dataset...")
        df = pd.read_csv("hf://datasets/regicid/LRFAF/corpus.csv")
        df_sample = df.sample(n=min(100, len(df)), random_state=42)
        print(f"Dataset loaded. Processing {len(df_sample)} songs.")
    except Exception as e:
        print(f"Error: {e}")
        exit(1)
    
    # Initialize LLM
    model_name = "Qwen/Qwen3-Next-80B-A3B-Instruct-FP8"
    llm = LLM(
        model=model_name,
        tensor_parallel_size=2,
        enforce_eager=True,
        max_model_len=8192,
        gpu_memory_utilization=0.98
    )
    
    # --- 3. CONFIGURE STRUCTURED OUTPUT ---
    structured_outputs_params = StructuredOutputsParams(
        json=InequalityAnalysis.model_json_schema()
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1024,
        structured_outputs=structured_outputs_params
    )

    # --- 4. SEQUENTIAL INFERENCE LOOP ---
    results = []
    print("Starting sequential analysis...")

    for i, (index, row) in enumerate(df_sample.iterrows()):
        # Identify text column (handles different CSV structures)
        song_text = next((row[c] for c in ['text', 'lyrics', 'paroles'] if c in row and isinstance(row[c], str)), "")
        
        if not song_text:
            continue

        messages = [
            {"role": "system", "con