from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser


# Load environment variables
load_dotenv()
prompt1=PromptTemplate(
    template="genrate a detailed report about {topic}\n",
    input_variables=['topic']
)
prompt2=PromptTemplate(
    template="genrate 5 pointer summary on given text {text}\n",
    input_variables=['text']
)

model=GoogleGenerativeAI(model='gemini-1.5-pro')
parser=StrOutputParser()

chain=prompt1| model| parser|prompt2| model |parser

result=chain.invoke({'topic':'unempolyment in pakistan'})
chain.get_graph().print_ascii()
print(result)