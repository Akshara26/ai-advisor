from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

system_prompt = """You are an academic advisor at a university. 
You help students with course selection, degree requirements, and academic planning.
Be specific, concise, and always ask clarifying questions if you need more context 
like the student's major, year, or completed courses."""

conversation_history = []

def chat(user_message):
    conversation_history.append({
        "role": "user",
        "content": user_message
    })
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": system_prompt}] + conversation_history
    )
    
    assistant_message = response.choices[0].message.content
    
    conversation_history.append({
        "role": "assistant",
        "content": assistant_message
    })
    
    return assistant_message

print("University Advisor - type 'quit' to exit\n")

while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break
    response = chat(user_input)
    print(f"\nAdvisor: {response}\n")

