from openai import OpenAI
import base64
import pandas as pd

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


client = OpenAI(
    # api_key="sk-or-vv-142b5cdef96bfbb8ff64d5efa42278ce15c60a67b8403757b0fce2200431bf87", # ваш ключ в VseGPT после регистрации
    api_key="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",
    # base_url="https://api.vsegpt.ru/v1",
    base_url="https://openrouter.ai/api/v1"
)

base64_image = encode_image("test_image.jpg")
messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "List all chemical compounds present in a figure, including reaction starting compounds, products and conditions above arrows."},
                {
                    "type": "image_url",
                    "image_url": f"data:image/jpeg;base64,{base64_image}",
                },
            ],
        }
    ]

results = []

for model in ['openai/gpt-5-chat', 'google/gemini-2.0-flash-lite-001', 'openai/gpt-4o']:
    response_big = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.8,
        n=1,
        max_tokens=2000,
        extra_headers={ "X-Title": "ChemicalChatBot" }
    )

    response = response_big.choices[0].message.content
    
    question_text = ""
    for msg in messages:
        if msg["role"] == "user":
            if isinstance(msg["content"], list):
                for block in msg["content"]:
                    if block["type"] == "text":
                        question_text += block["text"] + " "
            else:
                question_text += msg["content"]

    results.append({
        "model": model,
        "question": question_text.strip(),
        "answer": response.strip()
    })

    print(f"\nModel: {model}")
    print(f"Question: {question_text.strip()}")
    print(f"Answer: {response.strip()}")
    
df = pd.DataFrame(results)

csv_filename = "model_responses.csv"
try:
    existing_df = pd.read_csv(csv_filename)
    df = pd.concat([existing_df, df], ignore_index=True)
except FileNotFoundError:
    pass

df.to_csv(csv_filename, index=False, encoding="utf-8")

print(f"\n✅ Results saved to {csv_filename}")