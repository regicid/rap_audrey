from vllm import LLM, SamplingParams
import os
import pandas as pd
from tqdm import tqdm
from datetime import date
import json  # Missing import
from pydantic import BaseModel, Field  # Added Field
from typing import List, Optional, Literal

new_model_name = "NousResearch/Hermes-4-14B"
llm = LLM(new_model_name)
sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=1000)
rappeurs = pd.read_csv("/home/decourson/rap_audrey/rappeurs_bio.csv")

class Bio(BaseModel):
    date_of_birth: Optional[date] = Field(
        None, 
        description="Date of birth in YYYY-MM-DD format. If only year is known, use YYYY-01-01."
    )
    parents_profession: Optional[str] = Field(None,
        description = "If mentioned, the social class or the job the parents.")
    country_of_origin: Optional[str] = Field(
        None, 
        description="His country of origin, not necessarily where he grew up, but what defines his or her ethnicity."
    )
    birth_city: Optional[str] = Field(
        None, 
        description="The city where he or she was born."
    )
    childhood_city: Optional[str] = Field(
        None, 
        description="The city where he grew up."
    )
    activity_city: Optional[str] = Field(
        None, 
        description="The city where he has lived for the most part during his or her rapper career."
    )
    gender: Optional[Literal["men", "women"]] = Field(
        None,
        description="Gender of the rapper: 'men' or 'women'"
    )
    education_level: Optional[str] = Field(
        None, 
        description="The education level of the rapper: whether he or she finished highschool (baccalauréat in France), and any university stuff he did. If it is not mentioned, put None. If it is mentioned that he did not study or did not finish highschool, mention it here."
    ) 

schema = Bio.model_json_schema()  # Changed from schema_json()

def prompter(text):
    tool_definition = {
        "type": "function",
        "function": {
            "name": "annotate_biographies",
            "description": "You extract infos about French rappers from their biographies and output a structured format.",
            "parameters": schema
        }
    }
    
    prompt = f"""<|im_start|>system
You are the annotator of French rappers biographies.
<tools>
{json.dumps(tool_definition)}
</tools><|im_end|>
<|im_start|>user
Extract informations from this biography of a French rapper. If the informations are not present, put NA or an empty string. 
Here is the content: {text}<|im_end|>
<|im_start|>assistant
"""
    return prompt

# Define files - this was missing!
files = os.listdir("lemonde_update/")

prompts = []
for file in tqdm(files):
    with open("lemonde_update/" + file, "r") as f:  # Better file handling
        text = f.read()
    prompts.append(prompter(text))

outputs = llm.generate(prompts, sampling_params)

# Extract actual text from outputs
bio_results = [output.outputs[0].text for output in outputs]
rappeurs["bio_json"] = bio_results

# Save to different file to avoid overwriting
rappeurs.to_csv("rappeurs_bio_annotated.csv", index=False)