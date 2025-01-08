from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI 
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, AIMessage
from langchain.tools.retriever import create_retriever_tool

from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS

# Create Retriever
loader = WebBaseLoader('https://python.langchain.com/docs/expression_language/')
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20
)

splitDocs = splitter.split_documents(docs)

embedding = OpenAIEmbeddings()
vectorStore = FAISS.from_documents(docs, embedding=embedding)

retriever = vectorStore.as_retriever(
    search_kwargs={
        "k": 2
    }
)

model = ChatOpenAI(
    model="gpt-3.5-turbo-1106",
    temperature=0.7
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Tu es un assistant amical appelé Max."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

tavilySearch = TavilySearchResults()
retriever_tool = create_retriever_tool(
    retriever,
    "lcel_search_tool",
    "Utilise cet outil lorsque tu dois rechercher de l'information sur LangChain Expression Language (LCEL)."
)

tools = [tavilySearch, retriever_tool]

agent = create_openai_functions_agent(
    llm=model,
    prompt=prompt,
    tools=tools
)

agentExecutor = AgentExecutor(
    agent=agent,
    tools=tools
)

def process_chat(agentExecutor, user_input, chat_history):
    response = agentExecutor.invoke({
        "input": user_input,
        "chat_history": chat_history
    })

    return response["output"]

if __name__ == '__main__':

    chat_history = []
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break

        response = process_chat(agentExecutor, user_input, chat_history)
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response))
        print("Assistant:", response)