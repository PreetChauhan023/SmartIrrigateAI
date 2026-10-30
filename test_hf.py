import traceback
from pathlib import Path

import streamlit as st
from huggingface_hub import InferenceClient

# Load configuration from Streamlit secrets
token = st.secrets["HF_TOKEN"]

model = st.secrets.get(
    "HF_LLM_REPO",
    "Qwen/Qwen3-0.6B"
)

print("=" * 50)
print("SMARTIRRIGATE - HUGGING FACE API TEST")
print("=" * 50)

print(f"\nModel: {model}")
print(f"Token detected: {bool(token)}")

try:
    client = InferenceClient(
        provider="featherless-ai",
        api_key=token,
        timeout=60
    )

    print("\nSending test question...")

    response = client.chat_completion(
        model=model,
        messages=[
            {
                "role": "user",
                "content": (
                    "Explain why data drift is important "
                    "for irrigation predictions in two sentences."
                )
            }
        ],
        max_tokens=150,
        temperature=0.2
    )

    print("\nSUCCESS! Hugging Face LLM is working.\n")
    print(response.choices[0].message.content)

except Exception:
    print("\nHUGGING FACE API ERROR:\n")
    traceback.print_exc()