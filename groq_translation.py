import os
from groq import Groq

# Charger les variables d'environnement depuis .env
from dotenv import load_dotenv
load_dotenv()

# Initialize the Groq client
client = Groq()

# Specify the path to the audio file
filename = os.path.dirname(__file__) + "/Input/IA.mp3" # Replace with your audio file!
print(f"Translating file: {filename}")

# Open the audio file
with open(filename, "rb") as file:
    # Create a translation of the audio file
    translation = client.audio.translations.create(
      file=(filename, file.read()), # Required audio file
      model="whisper-large-v3", # Required model to use for translation
      prompt="Specify context or spelling",  # Optional
      # language="en", # Optional ('en' only)
      response_format="text",  # Optional
      temperature=0.0  # Optional
    )
    # Print the translation text
    print(translation)