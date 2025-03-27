from langchain_openai import ChatOpenAI 
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain_community.chat_message_histories.upstash_redis import UpstashRedisChatMessageHistory

import config  # on importe les variables depuis config.py

def create_chat_chain(session_id: str = "chat2"):
    # Instancier l'historique
    history = UpstashRedisChatMessageHistory(
        url=config.UPSTASH_REDIS_URL,
        token=config.UPSTASH_REST_TOKEN,
        session_id=session_id
    )

    # Créer le modèle
    model = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.7
    )

    # Créer le prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Tu es un assistant amical appelé Max."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    # Créer la mémoire
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        chat_memory=history
    )

    # Créer la chain
    chain = LLMChain(
        llm=model,
        prompt=prompt,
        memory=memory,
        verbose=True
    )

    return chain
