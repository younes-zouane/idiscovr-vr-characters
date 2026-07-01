import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.environ["DEEPINFRA_API_KEY"],
    base_url="https://api.deepinfra.com/v1/openai",
)

MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"

def ask(message):
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=300,
        messages=[{"role": "user", "content": message}]
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    while True:
        user_input = input("You: ")
        if user_input.lower() in ("quit", "exit"):
            break
        reply = ask(user_input)
        print("AI:", reply)