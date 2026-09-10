import os
import ollama


def get_client():
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    return ollama.Client(host=host)


def generate(model: str, prompt: str, system: str = None, temperature: float = 0.0) -> str:
    """
    Single-shot generation. temperature=0.0 by default since SQL generation
    should be deterministic, not creative.
    """
    client = get_client()
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = client.chat(
        model=model,
        messages=messages,
        options={"temperature": temperature},
    )
    return response["message"]["content"]
