from dotenv import load_dotenv
import random
import json
import os
from openai import OpenAI
import pandas as pd

# Load environment variables
load_dotenv()
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)
for i in range(2):
    print("###########,",i,",############")
        
    def encode_prompt(prompt_instructions):
        """Encode multiple prompt instructions into a single string."""
        prompt = open("./prompt.txt").read() + "\n"

        for idx, task_dict in enumerate(prompt_instructions):
            USER_COMMAND, instruction, input, output = task_dict["USER COMMAND"], task_dict["INSTRUCTIONS"], task_dict["INPUT"], task_dict["OUTPUT"]
            prompt += f"###\n"
            prompt += f"{idx + 1}. USER COMMAND: {USER_COMMAND}\n"
            prompt += f"{idx + 1}. INSTRUCTIONS: {instruction}\n"
            prompt += f"{idx + 1}. INPUT: {input}\n"
            prompt += f"{idx + 1}. XML BEHAVIOR TREE OUTPUT:\n{output}\n"
        # prompt += f"###\n"
        # prompt += f"{idx + 2}. USER COMMAND:"
        return prompt

    # Open and read the JSON file
    with open("seed task.json", 'r') as file:
        data = json.load(file)

    seed_instruction_data = [{"USER COMMAND": t["USER COMMAND"], "INSTRUCTIONS": t["INSTRUCTIONS"], "INPUT": t["INSTANCES"][0]["INPUT"], "OUTPUT": t["INSTANCES"][0]["OUTPUT"]} for t in data]
    # print(f"Loaded {len(seed_instruction_data)} human-written seed instructions")

    batch_inputs = random.sample(seed_instruction_data, 2)
    # print(len(batch_inputs))

    prompt_tamplet = encode_prompt(batch_inputs)

    with open("prompt_22.txt", "w") as file:
        file.write(prompt_tamplet)

    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt_tamplet,
        temperature=1,
        max_tokens=1024*3,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    # print(response)

    text = response.choices[0].text
    # print(text)

    with open("data_generated_2.txt", "a") as file:
        file.write(text)
