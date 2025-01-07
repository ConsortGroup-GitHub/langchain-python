from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains import create_retrieval_chain

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains.history_aware_retriever import create_history_aware_retriever


def get_documents_from_web(url):
    loader = WebBaseLoader(url)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=20
    )

    splitDocs = splitter.split_documents(docs)
    return splitDocs

def create_db(docs):
    embedding = OpenAIEmbeddings()
    vectorStore = FAISS.from_documents(docs, embedding=embedding)
    return vectorStore

def create_chain(vectorStore):
    # Instantiate Model
    model = ChatOpenAI(
        model="gpt-3.5-turbo-1106",
        temperature=0.4
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Réponds STP à la question de l'utilisateur en te basant sur le contexte : {context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    parser = StrOutputParser()

    # chain = prompt | model | parser
    chain = create_stuff_documents_chain(
        llm=model,
        prompt=prompt,
        output_parser=parser
    )

    retriever = vectorStore.as_retriever(
        search_kwargs={
            "k": 3
        }
    )

    retriever_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        ("human", "En te basant sur la conversation ci-dessus, génère une requête de recherche pour obtenir l'information pertinente par rapport à la conversation.")
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm=model,
        retriever=retriever,
        prompt=retriever_prompt
    )

    retrieval_chain = create_retrieval_chain(
        # retriever,
        history_aware_retriever,
        chain
    )

    return retrieval_chain

def process_chat(chain, question, chat_history):
    response = chain.invoke({
        "input": question,
        "context": docs,
        "chat_history": chat_history
    })

    return response["answer"]


if __name__ == '__main__':
    docs = get_documents_from_web('https://python.langchain.com/docs/expression_language/')
    vectorStore = create_db(docs)
    chain = create_chain(vectorStore)

    chat_history = []
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break

        response = process_chat(chain, user_input, chat_history)
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response))
        print("Assistant:", response)