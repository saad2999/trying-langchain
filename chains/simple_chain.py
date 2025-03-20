from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser


# Load environment variables
load_dotenv()
prompt=PromptTemplate(
    template="genrate 5 interesting facts about {topic}\n facts should be random on every run",
    input_variables=['topic']
)

model=GoogleGenerativeAI(model='gemini-1.5-pro')
Parser=StrOutputParser()

chain=prompt|model|Parser
result=chain.invoke({'topic':'books reading'})
print(result)
chain.get_graph().print_ascii()
