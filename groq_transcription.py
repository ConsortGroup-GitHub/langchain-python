import os
import json
from groq import Groq

# Charger les variables d'environnement depuis .env
from dotenv import load_dotenv
load_dotenv()

# Initialize the Groq client
client = Groq()

# Specify the path to the audio file
filename = os.path.dirname(__file__) + "/Input/IA.mp3" # Replace with your audio file!$
print(f"Transcribing file: {filename}")

# Open the audio file
with open(filename, "rb") as file:
    # Create a transcription of the audio file
    transcription = client.audio.transcriptions.create(
      file=file, # Required audio file
      model="whisper-large-v3-turbo", # Required model to use for transcription
      prompt="Specify context or spelling",  # Optional
      response_format="verbose_json",  # Optional
      timestamp_granularities = ["word", "segment"], # Optional (must set response_format to "json" to use and can specify "word", "segment" (default), or both)
      language="fr",  # Optional
      temperature=0.0  # Optional
    )
    # To print only the transcription text, you'd use print(transcription.text) (here we're printing the entire transcription object to access timestamps)
    print(json.dumps(transcription, indent=2, default=str))