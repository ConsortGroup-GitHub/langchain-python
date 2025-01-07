from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Instantiate Model
llm = ChatOpenAI(
    model="gpt-3.5-turbo-1106",
    temperature=0.2
)

# Prompt Template
# prompt = ChatPromptTemplate.from_template("Raconte moi une histoire à propos d'un {topic}")
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Génère une liste de 10 synonymes du mot suivant. Retourne le résultat sous forme de liste, avec des virgules comme séparateurs."),
        ("human", "{input}")
    ]
)

# Create LLM Chain
chain = prompt | llm

response = chain.invoke({"input": "content"})
print(response.content)