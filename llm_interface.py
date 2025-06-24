#!/usr/bin/env python3 

from abc import ABC, abstractmethod
from dotenv import load_dotenv
import os
from groq import Groq
import requests
import ollama 
import pandas as pd
# Load from .env file
load_dotenv()

groq_key = os.getenv("GROQ_API_KEY")

import os
from groq import Groq
from abc import ABC, abstractmethod

# === Abstract Base Interface ===
class LLMInterface(ABC):
    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def run(self, prompt: str) -> str:
        pass


# === LlamaModel using Groq ===
class LlamaModel(LLMInterface):
    def setup(self):
        self.name = "llama3-70b-8192"
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY is not set in the environment.")
        self.client = Groq(api_key=api_key)

    def run(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.name
        )
        return response.choices[0].message.content.strip()

class DeepSeekR1Model(LLMInterface):
    def setup(self):
        self.name = "deepseek-r1:1.5b"  # or "deepseek-llm:chat"
        self.api_url = "http://localhost:11434/api/generate"

    def run(self, prompt: str) -> str:
        result = ollama.generate(model=self.name, prompt=prompt)
        print(result['response'])

# === Manager to Use All Models ===
class LLMManager:
    def __init__(self):
        self.models = []
        self.responses = {}

    def register_model(self, model_instance: LLMInterface):
        model_instance.setup()
        self.models.append(model_instance)
        self.responses[model_instance.name] = []

    def run_models(self, df: pd.DataFrame, prompt_column: str = "prompt") -> pd.DataFrame:
        """
        Run each prompt in the DataFrame through all registered models,
        and add each model's response as a new column.

        Returns the updated DataFrame.
        """
        for model in self.models:
            model_name = model.name
            print(f"Running model: {model_name}")
            responses = []

            for i, row in df.iterrows():
                try:
                    prompt = row[prompt_column]
                    response = model.run(prompt)
                except Exception as e:
                    print(f"[Error row {i}, model {model_name}]: {e}")
                    response = "ERROR"
                responses.append(response)

            df[model_name + "_response"] = responses

        return df
        


def main():
    ds = DeepSeekR1Model()
    ds.setup()
    ds.run("Give another word for cat?")
    

if __name__ == "__main__":
    main()