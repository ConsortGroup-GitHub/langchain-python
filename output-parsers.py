from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, CommaSeparatedListOutputParser, JsonOutputParser
# from langchain_core.pydantic_v1 import BaseModel, Field


# Instantiate Model
llm = ChatOpenAI(
    model="gpt-3.5-turbo-1106",
    temperature=0.7
)

def call_string_output_parser():
    # Prompt Template
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Raconte moi STP une histoire courte et drôle sur le mot suivant : "),
            ("human", "{input}")
        ]
    )

    parser = StrOutputParser()

    # Create LLM Chain
    chain = prompt | llm | parser

    return chain.invoke({"input": "panda"})



# def call_list_output_parser():
#     # Prompt Template
#     prompt = ChatPromptTemplate.from_messages(
#         [
#             ("system", "Génère une liste de 10 synonymes du mot suivant. Retourne le résultat sous forme de liste, avec des virgules comme séparateurs."),
#             ("human", "{input}")
#         ]
#     )

#     parser = CommaSeparatedListOutputParser()

#     # Create LLM Chain
#     chain = prompt | llm | parser

#     return chain.invoke({"input": "triste"})


# def call_json_output_parser():
#     # Prompt Template
#     prompt = ChatPromptTemplate.from_messages(
#         [
#             ("system", "Extrais de l'information de la phrase suivante. \nInstructions de format: {format_instructions}"),
#             ("human", "{phrase}")
#         ]
#     )

#     class Person(BaseModel):
#         name: str = Field(description="Le nom de la personne.")
#         age: int = Field(description="L'âge de la personne.")


#     parser = JsonOutputParser(pydantic_object=Person)

#     # Create LLM Chain
#     chain = prompt | llm | parser

#     return chain.invoke({"phrase": "Max est un garçon qui a 30 ans."})



print(call_string_output_parser())
# print(call_list_output_parser())
# print(call_json_output_parser())