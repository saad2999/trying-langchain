from google import genai
from pydantic import BaseModel
import os
from dotenv import load_dotenv
load_dotenv()
from pydantic import BaseModel, Field
from typing import Optional, Literal

class Review(BaseModel):
    key_themes: list[str] = Field(description="Write down all the key themes discussed in the review in a list")
    summary: str = Field(description="A brief summary of the review")
    sentiment: Literal["pos", "neg"] = Field(description="Return sentiment of the review either negative, positive or neutral")
    pros: Optional[list[str]] = Field(default=None, description="Write down all the pros inside a list")
    cons: Optional[list[str]] = Field(default=None, description="Write down all the cons inside a list")
    name: Optional[str] = Field(default=None, description="Write the name of the reviewer")


api_key = os.getenv("GENAI_API_KEY")


client = genai.Client(api_key=api_key)
response = client.models.generate_content(
    model='gemini-2.0-flash',
    contents=""""I recently bought the Pixel 8 Pro, and it’s been a fantastic experience overall! The camera is outstanding, capturing sharp and vibrant photos even in low light. The AI-powered photo editing features are a game-changer. Performance-wise, the Tensor G3 chip handles everything smoothly, from gaming to multitasking. The battery life is decent, lasting a full day with moderate use. 

However, the phone tends to heat up a bit during heavy gaming sessions, and the lack of a microSD slot is disappointing. Also, the price feels a bit steep for what it offers compared to competitors.

Pros:
- Excellent camera quality
- Smooth performance
- AI photo editing is impressive

Cons:
- Tends to heat up
- No microSD slot
- Pricey

Review by Alex Turner
 """,
    config={
        'response_mime_type': 'application/json',
        'response_schema': list[Review],
    },
)
# Use the response as a JSON string.
print(response.text)

# Use instantiated objects.
my_review: list[Review] = response.parsed