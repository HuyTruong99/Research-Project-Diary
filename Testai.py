
from openai import OpenAI

try:
    client = OpenAI()

    print("Connecting to OpenAI...")

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "user",
                "content": "Reply exactly: LLM CONNECTION SUCCESSFUL"
            }
        ]
    )

    print("SUCCESS!")
    print("LLM Response:", response.choices[0].message.content)

except Exception as e:
    print("CONNECTION FAILED!")
    print("Error:", e)
