import httpx
from langchain_openai import ChatOpenAI
import openai

# Charger les variables d'environnement depuis .env
from dotenv import load_dotenv
load_dotenv()

# Désactiver la vérification SSL pour OpenAI
openai.verify_ssl_certs = False

# Initialisation du modèle OpenAI avec httpx pour ignorer SSL
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7,
    http_client=httpx.Client(verify=False)  # Ignore la vérification SSL
)

# Tester une requête simple
response = llm.invoke("Bonjour ! Peux-tu STP me raconter une blague sur des chiens ?")
print(response)
