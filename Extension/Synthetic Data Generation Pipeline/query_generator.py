from google import genai
from google.genai.types import HttpOptions, ModelContent, Part, UserContent
import json
import os
import timeit
import argparse
import time

query_templates = [
    "1. Where is object X before / after event Y?",
    "2. Where is object X?",
    "3. What did I put in X?",
    "4. How many X’s? (quantity question)",
    "5. What X did I Y?",
    "6. In what location did I see object X ?",
    "7. What X is Y?",
    "8. State of an object",
    "9. Where is my object X?",
    "10. Where did I put X?",
    "11. Who did I interact with when I did activity X?",
    "12. Who did I talk to in location X?",
    "13. When did I interact with person with role X?",
]

def split_dict(d, n):
    """Split a dictionary `d` into `n` chunks."""
    items = list(d.items())
    k, m = divmod(len(items), n)
    return [dict(items[i * k + min(i, m):(i + 1) * k + min(i + 1, m)]) for i in range(n)]

# Read the config file
with open("config.json", 'r', encoding='utf-8') as f:
    conf = json.load(f)  # Load JSON data

# Set the environment variables required to request to genai API
os.environ["GOOGLE_CLOUD_PROJECT"] = conf["GOOGLE_CLOUD_PROJECT"]
os.environ["GOOGLE_CLOUD_LOCATION"] = conf["GOOGLE_CLOUD_LOCATION"]
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = conf["GOOGLE_GENAI_USE_VERTEXAI"]
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = conf["GOOGLE_APPLICATION_CREDENTIALS"]

narration_path = conf["narration_path"]

with open(narration_path, 'r', encoding='utf-8') as f:
    ego4d_narrations = json.load(f)  # Load JSON data

import re
def extract_json_from_response(response_text):
    # Extract JSON object using regex
    match = re.search(r'\{.*\}', response_text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found.")
    
    json_str = match.group(0)
    return json.loads(json_str)
def generate_content(data_block_indx) -> str:

    narration_parts = split_dict(ego4d_narrations, 20)
    data_part = narration_parts[data_block_indx]

    client = genai.Client(http_options=HttpOptions(api_version="v1"))

    # Step 1: Send the guide as the context (only once)
    guide_text = open("llm_prompt.txt", "r", encoding="utf-8").read()
    
    init_guide = f"This is a guide you should keep in mind and work based on what it wants: \n{guide_text}"
    chat_session = client.chats.create(
        # Teach the annotation guide to LLM
        model="gemini-2.0-flash-001",
        history=[
            UserContent(parts=[Part(text=init_guide)]),
            ModelContent(
                parts=[Part(text="It seems it's for ego4d dataset. yes. Sure! I will send the json format you want for each group of narrations you send.")],
            ),
        ],
    )

    partition_size = len(data_part.keys())
    count = 0
    for video_indx, video_id in enumerate(data_part):
        if video_indx % 30 == 0:  # or every 10, or per video_id
            chat_session = client.chats.create(
            model="gemini-2.0-flash-001",
            history=[
                UserContent(parts=[Part(text=init_guide)]),
                ModelContent(parts=[Part(text="yeah, sure! It seems it's for ego4d dataset. I will send the json format you want for each group of narrations you send.")]),
            ],
    )
        for clip_indx, narration in enumerate(data_part[video_id]["narrations"]):
            start = timeit.default_timer()
            nar_string = "\n".join(narration["texts"])
            print("nar_string", nar_string)
            while True:
                try:
                    message = f"""
                    You are given a short sequence of 5 narrations that describe what a person is doing in a first-person (ego-centric) video.

                    Your job is to generate a single **natural language question** that:
                    - Matches the intent of one of the 13 templates listed below
                    - Can be **clearly answered by watching the video segment described by these narrations only**
                    - Refers only to **visible and interacted objects** — avoid large immovable entities like houses, buildings, mountains, cities, etc.
                    - Is written in **past tense**
                    - Is **specific**, **grounded**, and **concrete** (e.g., “Where did I place the cup?” is good, “Where was I staring?” is bad)
                    - **Rephrases the template naturally** — do NOT copy the exact words or verbs in the template; use synonyms and rewrite it to suit the narration
                    - Avoids general or vague questions (like "What happened?" or "What was I doing?")
                    - Avoids making assumptions about parts of the video **outside** the given 5 narrations
                    - All queries must be in past tense (e.g., "was", "did", "had", "saw")
                    You don't have to use exactly the template. You have flexibility to rephrase and reverb, but in meaningful ways and the answer shoule be foundable in the video.
                    Only return a JSON object in this format:

                    {{
                    "template": <template number>,
                    "query": "<natural language question>"
                    }}

                    Templates:
                    {chr(10).join([f"{i+1}. {t}" for i, t in enumerate(query_templates)])}

                    Narrations:
                    {nar_string}
                    """
                    response = chat_session.send_message(message).text
                    break  # Exit the loop if successful
                except Exception as e:
                    if "resource exhausted" in str(e).lower():
                        print("Resource exhausted, waiting 1 second before retrying...")
                        time.sleep(1)
                    else:
                        raise  # Re-raise unexpected exceptions
            # Remove the Markdown code block markers
            if response.startswith("```"):
                # Remove the starting and ending triple backticks and optional language label
                response = response.strip("`").split('\n', 1)[-1].rsplit('\n', 1)[0]

            print(f"Response (raw): {repr(response)}")
            # response_dict = json.loads(response.replace("json", ""))
            response_dict = extract_json_from_response(response)
            narration["nlq_query"] = response_dict["query"]
            narration["template"] = query_templates[int(response_dict["template"])-1]
            print("text:", nar_string)
            print("query:", response)
            stop = timeit.default_timer()
            print('Time: ', stop - start)
            print("-----------------------------------------------")
        count += 1
        print(str(count)+"/"+str(partition_size))
    with open(f"queries_part{data_block_indx}.json", "w") as json_file:
        json.dump(data_part, json_file, indent=4)
    return response

if __name__ == "__main__":

    parser = argparse.ArgumentParser("simple_example")
    parser.add_argument("data_part_index", help="The index of a specific data block.", type=int)

    args = parser.parse_args()
    generate_content(args.data_part_index)