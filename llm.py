from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7
)

response = llm.invoke("Bonjour ! Peux-tu STP me raconter une blague sur des chiens ?")
print(response)