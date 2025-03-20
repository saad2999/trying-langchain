from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field


# Load environment variables
load_dotenv()

model=GoogleGenerativeAI(model='gemini-1.5-pro')

class Person(BaseModel):
    name:str=Field(description="name of the person")
    age:int=Field(gt=18,description="age of the person")
    city:str=Field(description="city of the person")
    
parser=PydanticOutputParser(pydantic_object=Person)
template=PromptTemplate(
    template="Give me the name age city of fictional {place} person.\n{format_instruction}",
    input_variables=['place'],
    partial_variables={"format_instruction": parser.get_format_instructions()}
)
prompt=template.invoke({'place':'Elfhame '})
result=model.invoke(prompt)
final_result=parser.parse(result)
print(final_result)