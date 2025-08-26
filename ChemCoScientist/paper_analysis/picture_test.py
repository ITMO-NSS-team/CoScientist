from openai import OpenAI
import base64

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
                {"type": "text", "text": "List all organic chemical structures present in a figure in the format of SMILES strings and IUPAC names."},
                {
                    "type": "image_url",
                    "image_url": f"data:image/jpeg;base64,{base64_image}",
                },
            ],
        }
    ]

for model in ['openai/gpt-5-chat']:
    response_big = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.8,
        n=1,
        max_tokens=2000,
        extra_headers={ "X-Title": "ChemicalChatBot" }
    )

    # print("Response BIG:",response_big)
    response = response_big.choices[0].message.content
    print("Response:",response)