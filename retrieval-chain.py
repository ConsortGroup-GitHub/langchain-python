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


def get_documents_from_web(url):
    loader = WebBaseLoader(url)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
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

    prompt = ChatPromptTemplate.from_template("""
        Réponds à la question de l'utilisateur :
        Contexte : {context}
        Question : {input}
    """)

    parser = StrOutputParser()

    # chain = prompt | model | parser
    chain = create_stuff_documents_chain(
        llm=model,
        prompt=prompt,
        output_parser=parser
    )

    retriever = vectorStore.as_retriever(
        search_kwargs={
            "k": 2
        }
    )

    retrieval_chain = create_retrieval_chain(
        retriever,
        chain
    )

    return retrieval_chain


docs = get_documents_from_web('https://python.langchain.com/docs/expression_language/')
vectorStore = create_db(docs)
chain = create_chain(vectorStore)


response = chain.invoke({
    "input": "C'est quoi, LCEL ?",
    "context": docs
})

print(response)