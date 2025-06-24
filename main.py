#! /usr/bin/env python3

import llm_interface
import pandas as pd


# def get_models():


# def response_generator(list_of_models, question, prompt):


# def store_response(question, prompt, list_of_responses, filename):



def main():
    llama = llm_interface.LlamaModel()
    llama.setup()
    deepseek = llm_interface.DeepSeekR1Model()
    deepseek.setup()

    manager = llm_interface.LLMManager()
    manager.register_model(llama)
    manager.register_model(deepseek)

    df = pd.read_csv("medmcqa_questions.csv")


    df_responses = manager.run_models(df)
    df_responses.to_csv("draft.csv", index=False)


if __name__ == '__main__':
    main()