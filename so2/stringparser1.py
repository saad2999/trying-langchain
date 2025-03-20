import os
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser


# Load environment variables
load_dotenv()

model=GoogleGenerativeAI(model='gemini-1.5-pro')

template1 = PromptTemplate.from_template("Write a detailed report on {topic}")
template2 = PromptTemplate.from_template("Write a 5-line summary on the following text:\n{text}")

parer=StrOutputParser()

chain=template1 | model | parer |template2| model|parer

result=chain.invoke({'topic':'black hole'})
print(result)