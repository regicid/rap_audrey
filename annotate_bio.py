from vllm import LLM, SamplingParams
import os
import pandas as pd
from tqdm import tqdm
from datetime import date
import sys
new_model_name = "NousResearch/Hermes-4-14B"

llm = LLM(new_model_name)
sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=1000)

rappeurs = pd.read_csv("rappeurs_bio.csv")

from pydantic import BaseModel
from typing import List, Optional, Literal

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
    education_level: Optional[str] = Optional[str] = Field(
        None, 
        description="The education level of the rapper: whether he or she finished highschool (baccalauréat in France), and any university stuff he did. If it is not mentioned, put None. If it is mentioned that he did not study or did not finish highschool, mention it here."
    ) 

schema = Bio.schema_json()


def prompter(text):
    tool_definition = {
        "type": "function",
        "function": {
            "name": "annotate_biographies",
            "description": "You extract infos about French rappers from their biographies and output a structured format.",
            "parameters": schema  # Your Pydantic schema converted to JSON Schema
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

prompts = []
for file in tqdm(files):
    file = open("lemonde_update/"+file,"r")
    text = file.read()
    file.close()
    prompts.append(prompter(text))

outputs = llm.generate(prompts, sampling_params)

rappeurs["bio_json"] = outputs


rappeurs.to_csv(f"rappeurs_bio.csv")
