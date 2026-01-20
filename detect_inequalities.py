import os
import pandas as pd
import json
from typing import List, Literal
from pydantic import BaseModel

# 1. Environment and vLLM Imports
os.environ["VLLM_USE_V1"] = "1"

from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams # New import for v0.13.0+

# --- 1. DEFINE OUTPUT SCHEMA ---
class InequalityMention(BaseModel):
    lines: str
    type: Literal["inégalités des chances", "inégalités raciales", "inégalités de genre", "autre"]
    stance: Literal[
        "reproduction", "dénonciation", "constat"   # the text criticizes / exposes the inequality
    ]
    justification: str
    confidence: Literal["faible", "moyenne", "élevée"]

class InequalityAnalysis(BaseModel):
    mentions: List[InequalityMention]

# --- 2. LOAD & SAMPLE DATA ---
try:
    print("Loading dataset...")
    df = pd.read_csv("hf://datasets/regicid/LRFAF/corpus.csv")
    df_sample = df.sample(n=min(100, len(df)), random_state=42)
    print(f"Dataset loaded. Processing {len(df_sample)} songs.")
except Exception as e:
    print(f"Error: {e}")
    exit(1)

# --- 3. INITIALIZE LLM ---
def main():
    model_name = "Qwen/Qwen3-Next-80B-A3B-Instruct-FP8"
    llm = LLM(
        model=model_name,
        tensor_parallel_size=2,
        enforce_eager=True,
        max_model_len=8192,
        gpu_memory_utilization=0.98
    )
    
    # ... rest of your code ...

if __name__ == '__main__':
    main()

# --- 4. CONFIGURE STRUCTURED OUTPUT ---
# This is the updated syntax for v0.12.0 and later
structured_outputs_params = StructuredOutputsParams(
    json=InequalityAnalysis.model_json_schema()
)

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=1024,
    structured_outputs=structured_outputs_params # Updated parameter
)

# --- 5. SEQUENTIAL INFERENCE LOOP ---
results = []
print("Starting sequential analysis...")

for i, (index, row) in enumerate(df_sample.iterrows()):
    # Identify text column (handles different CSV structures)
    song_text = next((row[c] for c in ['text', 'lyrics', 'paroles'] if c in row and isinstance(row[c], str)), "")
    
    if not song_text:
        continue

    messages = [
        {"role": "system", "content": """
        Tu es un annotateur expert. Analyse les inégalités dans les paroles et réponds en JSON. N’invente rien. Ne crée une entrée dans 'mentions' que lorsqu’une inégalité est clairement et explicitement exprimée dans le texte. S’il n’y a aucune inégalité explicite dans le morceau, retourne 'mentions': []. N’ajoute pas d’entrées pour expliquer l’absence d’inégalités. Dans la justification, explique le lien avec le contexte global
du morceau ou avec les lignes précédentes lorsque c’est pertinent. Avant d’annoter des lignes individuelles, identifie mentalement
le thème général du morceau. Si c'est pertinent, utilise ce thème pour justifier les mentions.

        Pour chaque mention :
        - indique le type d’inégalité
        - indique la stance 
        - indique un niveau de confidence :
          * "élevée", "moyenne", ou "faible"

        Définition de la stance :
        - "reproduction" : le texte reprend, normalise ou valorise une inégalité sans distance critique.

        - "constat" :
          le texte constate l’existence d’une inégalité ou d’une situation injuste sans l’approuver ni la condamner explicitement.

        - "dénonciation" :
          le texte critique l’inégalité, la condamne ou en souligne l’injustice.

        Définitions des types d’inégalités :

        - "inégalités de genre" : différences de traitement, de statut ou de valeur explicitement liées au sexe ou au genre (femmes, hommes, rôles genrés, sexisme).

        - "inégalités raciales" : discriminations ou hiérarchies fondées sur l’origine ethnique, la couleur de peau, la nationalité ou la race.

        - "inégalités des chances" :inégalités liées au milieu social, à la pauvreté, à l’accès à l’éducation, au travail ou aux ressources.

         # CRITÈRES D'EXCLUSION (CE QU'IL NE FAUT PAS ANNOTER) :
            - La souffrance personnelle vague : "Je suis triste", "J'erre sans but", "Je suis seul". Ce sont des émotions, pas des inégalités sociales. N'annote que si la souffrance est reliée à une cause sociale (ex: "Je suis triste car je n'ai pas de papiers").
            - La sexualité explicite consensuelle : Décrire un acte sexuel n'est pas une "inégalité de genre" sauf si le vocabulaire est dégradant, violent ou force la soumission.
            - La simple mention des catégories : Dire "les hommes et les femmes" n'est pas une reproduction d'inégalité. Idem sur les "nègres".

        """},
        {"role": "user", "content": f"Paroles :\n{song_text}"}
    ]

    # Use the tokenizer directly to avoid the 'unsafe' warning
    # apply_chat_template with tokenize=True returns token IDs
    prompt_token_ids = llm.get_tokenizer().apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True
    )

    try:
        # We pass prompt_token_ids instead of a string
        outputs = llm.generate(
        [{"prompt_token_ids": prompt_token_ids}],
        sampling_params=sampling_params
        )
 
 
        generated_text = outputs[0].outputs[0].text
        print(generated_text)
        results.append({
            "index": index,
            "analysis": json.loads(generated_text)
        })
        print(f"[{i+1}/100] Song {index} analyzed.")
    except Exception as e:
        print(f"Error at index {index}: {e}")

# --- 6. SAVE ---
with open("results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("Finished. Results saved to results.json.")
