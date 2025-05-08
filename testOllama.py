import requests

def test_ollama_chat(prompt):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "mistral",  # or llama2, gemma, etc. if you've pulled them
                "prompt": prompt,
                "stream": False
            }
        )
        data = response.json()
        return data["response"]
    except Exception as e:
        return f"Error: {e}"

# Test prompt
question = "What is overfitting in machine learning?"
answer = test_ollama_chat(question)

print("🧠 Question:", question)
print("🤖 Answer:", answer)
