import os
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.output_parsers import JsonOutputParser

# Load environment variables
load_dotenv()

# Initialize Google Gemini Model
model = GoogleGenerativeAI(model="gemini-1.5-pro")

# Initialize JSON Parser
parser = JsonOutputParser()

# Define Correct Prompt Template
template1 = PromptTemplate(
   template="Give me 5 facts about {topic}.\n{format_instruction}",
    input_variables=['topic'],
    partial_variables={"format_instruction": parser.get_format_instructions()}
)

# Format the prompt
# prompt = template1.format()

# response = model.invoke(prompt)
# fr=parser.parse(response)
# print(fr)
# print(type(fr))

chain=template1 | model | parser
result=chain.invoke({'topic': "random"})
print(str(result))
