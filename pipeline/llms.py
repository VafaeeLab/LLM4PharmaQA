412#!/usr/bin/env python3 

from abc import ABC, abstractmethod
from dotenv import load_dotenv
import os
from groq import Groq
import requests
import ollama 
import pandas as pd
import json
from google import genai
from google.genai import types
from huggingface_hub import InferenceClient
from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import OpenAI

# Load keys from .env file
load_dotenv()
groq_key = os.getenv("GROQ_API_KEY")
gemini_key = os.getenv("GOOGLE_AI_STUDIO_API_KEY")
huggingface_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")


class LLMInterface(ABC):
    @property
    def model_name(self):
        pass
    
    @property
    def model_id(self):
        pass
    
    @property
    def model_type(self):
        pass

    @property
    def model_family(self):
        pass
    
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def run(self, prompt: str) -> str:
        pass

# === Manager to Use All Models ===
class LLMManager:
    def __init__(self):
        self.models = []
        self.responses = {}

    def register_model(self, model_instance: LLMInterface):
        self.models.append(model_instance)
        self.responses[model_instance._model_name] = []

    def run_models(
        self,
        df: pd.DataFrame,
        prompt_column: str = "prompt"
    ) -> pd.DataFrame:
        """
        For each model:
        • send the prompt to every row
        • store the raw response text

        Columns created per model - f"{model._model_name}_response", "raw"
        """
        if not isinstance(df.columns, pd.MultiIndex):
            df.columns = pd.MultiIndex.from_tuples([(c, "") for c in df.columns])

        for m in self.models:
            df[(f"{m._model_name}_response", "raw")] = ""
            for idx, (i, row) in enumerate(df.iterrows()):
                prompt = (
                    row[(prompt_column, "")]
                    if (prompt_column, "") in df.columns
                    else row[prompt_column]
                )
                try:
                    raw = m.run(prompt)
                except Exception as e:
                    raw = f"ERROR: {e}"
                df.at[i, (f"{m._model_name}_response", "raw")] = raw

        return df

        

# === Llama3-8B using Groq ===
class Llama3_8B_Model(LLMInterface):
    _model_name = "llama3-8b"
    _model_id = "llama-3.1-8b-instant"
    _model_type = "foundation"
    _model_family = "llama"

    def __init__(self):
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY is not set in the environment.")
        self.client = Groq(api_key=api_key)

    def run(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self._model_id,
            temperature=1.0
        )
        content = response.choices[0].message.content
        if content is None:
            raise ValueError(f"Model {self._model_name} returned None content")
        return content.strip()


# === Llama3-70B using Groq ===
class Llama3_70B_Model(LLMInterface):
    _model_name = "llama3.3-70b"
    _model_id = "llama-3.3-70b-versatile"
    _model_type = "foundation"
    _model_family = "llama"

    def __init__(self):
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY is not set in the environment.")
        self.client = Groq(api_key=api_key)

    def run(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self._model_id,
            temperature=1.0
        )
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("Model returned None content")
        return content.strip()



# === Gemini 2.5 Flash Thinking ===
class Gemini25FlashNonThinking(LLMInterface):
    _model_name = "gemini-2.5-flash-nonthinking"
    _model_id = "gemini-2.5-flash"
    _model_type = "foundation"
    _model_family = "gemini"

    def __init__(self):
        self.client = genai.Client()

    def run(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self._model_id, 
            contents=prompt, 
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=0)
                # Turn off thinking:
                # thinking_config=types.ThinkingConfig(thinking_budget=0)
                # Turn on dynamic thinking:
                # thinking_config=types.ThinkingConfig(thinking_budget=-1)
            ),
        )
        if response.text is None:
            raise ValueError("Model returned None text")
        return response.text


class Gemini25FlashThinking(LLMInterface):
    _model_name = "gemini-2.5-flash-nonthinking"
    _model_id = "gemini-2.5-flash"
    _model_type = "reasoning"
    _model_family = "gemini"


    def __init__(self):
        self.client = genai.Client()

    def run(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self._model_id, 
            contents=prompt, 
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=-1)
                # Turn off thinking:
                # thinking_config=types.ThinkingConfig(thinking_budget=0)
                # Turn on dynamic thinking:
                # thinking_config=types.ThinkingConfig(thinking_budget=-1)
            ),
        )
        if response.text is None:
            raise ValueError("Model returned None text")
        return response.text

class Gemini25Pro(LLMInterface):
    _model_name = "gemini-2.5-pro-nonthinking"
    _model_id = "gemini-2.5-pro"
    _model_type = "reasoning"
    _model_family = "gemini"

    def __init__(self):
        self.client = genai.Client()

    def run(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self._model_id, 
            contents=prompt, 
        )
        
        if response.text is None:
            raise ValueError("Model returned None text")
        return response.text



# === OpenBio 8B using HuggingFace ===
class OpenBio8BModel(LLMInterface):

    _model_name = "openbio8B"
    _model_id = "aaditya/Llama3-OpenBioLLM-8B"
    _model_type = "fine-tuned"
    _model_family = "openbio"

    def __init__(self):
        self.client =  InferenceClient(
    provider="featherless-ai",
    api_key=os.environ["HF_TOKEN2"],
)

    def run(self, prompt: str) -> str:
        response = self.client.text_generation(
    model=self._model_id,
    prompt=prompt,
    max_new_tokens=256,
    do_sample=False,
)
        return response.strip() 

class OpenBio70BModel(LLMInterface):
    _model_name = "openbio70B"
    _model_id = "aaditya/Llama3-OpenBioLLM-70B"
    _model_type = "fine-tuned"
    _model_family = "openbio"
    

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self._model_id)
        self.model = AutoModelForCausalLM.from_pretrained(self._model_id)

    def run(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if text.startswith(prompt):
            text = text[len(prompt):].strip()
        return text

class Med42_8B_Model(LLMInterface):
    _model_name = "med42-8b"
    _model_id = "m42-health/Llama3-Med42-8B:featherless-ai"
    _model_type = "fine-tuned"
    _model_family = "med42"

    def __init__(self):
        self.client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=os.environ["HF_TOKEN2"],
        )   

    def run(self, prompt: str) -> str:
        completion = self.client.chat.completions.create(
            model=self._model_id,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=1.0,
        )
        return completion.choices[0].message.content.strip()

class Med42_70B_Model(LLMInterface):
    _model_name = "med42-70b"
    _model_id = "m42-health/Llama3-Med42-70B:featherless-ai"
    _model_type = "fine-tuned"
    _model_family = "med42"

    def __init__(self):
        self.client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_TOKEN2"],
)

    def run(self, prompt: str) -> str:
        completion = self.client.chat.completions.create(
            model=self._model_id,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=1.0,
        )
        return completion.choices[0].message.content.strip()

from ollama import chat
from ollama import ChatResponse

class DeepSeekR1_1_5_Model(LLMInterface):
    _model_name = "deepseek-r1-1.5"
    _model_id = "deepseek-r1:1.5b"
    _model_type = "reasoning"
    _model_family = "deepseek"

    def __init__(self):
        pass

    def run(self, prompt: str) -> str:
        # Create a chat completion using the Ollama Python client
        response: ChatResponse = chat(
            model=self._model_id,
            messages=[
                {"role": "user", "content": prompt}
            ],
            options={
                "temperature": 1.0,
            }
        )
        # Ollama's ChatResponse has .message and .message['content']
        return response.message["content"].strip()

if __name__ == "__main__":
    model_classes = [
        # Llama3_8B_Model,
        # Llama3_70B_Model,
        # Gemini25FlashNonThinking,
        # Gemini25FlashThinking,
        # Gemini25Pro,
        # Med42_8B_Model,
        # Med42_70B_Model,
        DeepSeekR1_1_5_Model,
    ]

    models = []
    for ModelClass in model_classes:
        try:
            models.append((ModelClass.__name__, ModelClass()))
        except Exception as e:
            models.append((ModelClass.__name__, f"ERROR: {e}"))

    prompt = "What is a cat?"
    outputs = []
    for m in models:
        name, instance = m
        if isinstance(instance, str) and instance.startswith("ERROR:"):
            outputs.append((name, instance))
            continue
        try:
            outputs.append((instance._model_name, instance.run(prompt)))
        except Exception as e:
            outputs.append((instance._model_name, f"ERROR: {e}"))

    for name, result in outputs:
        print(f"{name}: {result}\n")



["ERROR: Error code: 400 - {'error': {'message': 'The model `llama3-8b-8192` has been decommissioned and is no longer supported. Please refer to https://console.groq.com/docs/deprecations for a recommendation on which model to use instead.', 'type': 'invalid_request_error', 'code': 'model_decommissioned'}}", 
"ERROR: Error code: 400 - {'error': {'message': 'The model `llama3-70b-8192` has been decommissioned and is no longer supported. Please refer to https://console.groq.com/docs/deprecations for a recommendation on which model to use instead.', 'type': 'invalid_request_error', 'code': 'model_decommissioned'}}", 
"A **cat** is a small, domesticated carnivorous mammal, widely kept as a pet.\n\nHere's a breakdown of what that means and some other key characteristics:\n\n*   **Small:** While sizes vary by breed, most cats are relatively small compared to other common pets or wild animals.\n*   **Domesticated:** This means humans have selectively bred them over thousands of years to live alongside us. They are distinct from their wild ancestors (like the African wildcat).\n*   **Carnivorous:** Cats are obligate carnivores, meaning they *must* eat meat to survive and thrive. Their digestive systems are designed for a meat-based diet.\n*   **Mammal:** Like humans, cats are warm-blooded, have fur, give birth to live young, and nurse their offspring with milk.\n*   **Pet:** They are one of the most popular companion animals globally, valued for their companionship, playful nature, and sometimes their mouse-hunting abilities.\n\n**Other defining characteristics:**\n\n*   **Agile and Flexible:** They have incredibly flexible spines and powerful leg muscles, allowing them to jump, climb, and maneuver in tight spaces.\n*   **Sharp Senses:** Cats have excellent night vision, acute hearing (they can hear higher frequencies than humans), and a keen sense of smell. Their whiskers (vibrissae) are also crucial sensory organs.\n*   **Retractable Claws:** Most cats can retract their claws into sheaths, keeping them sharp and preventing them from getting dull during normal movement.\n*   **Unique Vocalizations:** They communicate through meows, purrs, hisses, growls, and other sounds.\n*   **Grooming:** Cats are meticulously clean animals, spending a significant portion of their day grooming themselves with their barbed tongues.\n*   **Independent but Affectionate:** While often seen as more independent than dogs, cats are capable of forming strong bonds with their human companions and showing affection.\n\nIn essence, a cat is a fascinating and complex creature that has woven itself into human society and hearts for millennia.", 
'A **cat** is a small, carnivorous mammal belonging to the family **Felidae**. When most people say "a cat," they are referring to the **domestic cat** (*Felis catus* or *Felis silvestris catus*), which is one of the most popular companion animals in the world.\n\nHere\'s a more detailed breakdown:\n\n1.  **Biological Classification:**\n    *   **Kingdom:** Animalia (animals)\n    *   **Phylum:** Chordata (vertebrates)\n    *   **Class:** Mammalia (mammals - warm-blooded, fur, live young, nurse with milk)\n    *   **Order:** Carnivora (carnivores - meat-eaters)\n    *   **Family:** Felidae (the cat family, which also includes wild cats like lions, tigers, leopards, etc.)\n    *   **Species:** *Felis catus* (domestic cat)\n\n2.  **Physical Characteristics:**\n    *   **Size:** Generally small to medium-sized, with variations across breeds.\n    *   **Coat:** Covered in fur, which comes in a vast array of colors, patterns (tabby, calico, tortie, solid), and lengths (short, medium, long).\n    *   **Head:** Typically roundish with prominent, often pointed, upright ears.\n    *   **Eyes:** Large, forward-facing, with excellent night vision and pupils that can constrict to a vertical slit in bright light.\n    *   **Whiskers:** Long, sensitive tactile hairs (vibrissae) on the muzzle, above the eyes, and on the forelegs, used for navigation and sensing their environment.\n    *   **Paws:** Padded paws with five toes on the front and four on the back.\n    *   **Claws:** **Retractable claws** (a key characteristic of most cats, allowing them to keep their claws sharp and hidden when not in use).\n    *   **Tail:** A long tail, crucial for balance.\n    *   **Teeth:** Specialized carnivorous teeth, including sharp canines for tearing and molars for shearing meat.\n\n3.  **Behavioral Characteristics:**\n    *   **Predatory:** Natural hunters, preying on small animals like rodents, birds, and insects. They are known for their stealth, agility, and powerful pounces.\n    *   **Grooming:** Extremely fastidious, spending a significant amount of time self-grooming.\n    *   **Vocalizations:** Communicate through a variety of sounds, including meows, purrs (a sign of contentment, but also stress), hisses, growls, trills, and chirps.\n    *   **Sleep:** Spend a large portion of their day sleeping (around 12-16 hours), often in short naps.\n    *   **Nocturnal/Crepuscular:** While they can adapt to human schedules, they are naturally most active at dawn and dusk.\n    *   **Playful:** Many cats, especially when young, are highly playful, engaging in mock hunting behaviors.\n    *   **Territorial:** They mark their territory through scent (glands on their face, paws, and urine).\n    *   **Social/Independent:** Domestic cats can form strong bonds with humans and other animals, but also retain a degree of independence.\n\n4.  **Relationship with Humans:**\n    *   **Domestication:** Cats were likely domesticated around 9,500 years ago, primarily for pest control (keeping rodents away from grain stores).\n    *   **Companionship:** Today, they are beloved pets, cherished for their affection, calming presence, and often amusing antics.\n    *   **Cultural Significance:** Cats hold various symbolic meanings across cultures, representing mystery, independence, luck, or even divinity.\n\nIn essence, a cat is a fascinating and diverse creature, ranging from a fierce wild predator to a cherished household companion, all sharing the distinctive traits of the Felidae family.',
 "ERROR: 503 UNAVAILABLE. {'error': {'code': 503, 'message': 'The model is overloaded. Please try again later.', 'status': 'UNAVAILABLE'}}", 
 'ERROR: Model aaditya/Llama3-OpenBioLLM-70B is not supported by provider nebius.']