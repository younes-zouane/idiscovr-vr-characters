import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.environ["DEEPINFRA_API_KEY"],
    base_url="https://api.deepinfra.com/v1/openai",
)

MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"

GENIE_SYSTEM_PROMPT = """You are the Genie of the lamp, straight out of the 
One Thousand and One Nights. You are loud, theatrical, and a bit of a 
show-off. You grant "wishes" by answering questions with flair and humor. 
Keep replies short (2-4 sentences) and full of personality. Never break 
character."""

# conversation memory: a running list of all messages
conversation_history = [
    {"role": "system", "content": GENIE_SYSTEM_PROMPT}
]

def ask_genie(message):
    conversation_history.append({"role": "user", "content": message})

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=300,
        messages=conversation_history
    )

    reply = response.choices[0].message.content
    conversation_history.append({"role": "assistant", "content": reply})
    return reply

if __name__ == "__main__":
    print("The Genie awaits your wishes... (type 'quit' to exit)\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ("quit", "exit"):
            break
        reply = ask_genie(user_input)
        print("Genie:", reply, "\n")