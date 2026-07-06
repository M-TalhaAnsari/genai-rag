import os
from ollama import chat
from gtts import gTTS
from IPython.display import Audio
import io

def generate_story(topic):

    prompt = f"""Write an engaging and educational story about {topic} for beginners. 
            Use simple and clear language to explain basic concepts. 
            Include interesting facts and keep it friendly and encouraging. 
            The story should be around 200-300 words and end with a brief summary of what we learned. 
            Make it perfect for someone just starting to learn about this topic."""
    

    response = chat(
        model="mistral",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )
    return response["message"]["content"]

topic = "the life cycle of butterflies"
story = generate_story(topic)


# Convert story to speech
tts = gTTS(story)

# Save the audio to a bytes buffer in memory
audio_bytes = io.BytesIO()
tts.write_to_fp(audio_bytes)
audio_bytes.seek(0)

Audio(audio_bytes.read(), autoplay=False)