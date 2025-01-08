import os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI 
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain_community.chat_message_histories.upstash_redis import UpstashRedisChatMessageHistory

history = UpstashRedisChatMessageHistory(
    url=os.getenv('UPSTASH_REDIS_URL'),
    token=os.getenv('UPSTASH_REST_TOKEN'),
    session_id="chat2"
)

model = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Tu es un assistant amical appelé Max."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    chat_memory=history
)

# chain = prompt | model
chain = LLMChain(
    llm=model,
    prompt=prompt,
    memory=memory,
    verbose=True
)

msg1 = {
    "input": "Mon nom est David."
}
resp1 = chain.invoke(msg1)

print(resp1)

msg2 = {
    "input": "Quel est mon nom ?"
}
resp2 = chain.invoke(msg2)

print(resp2)