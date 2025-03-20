from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI
from langchain.output_parsers import StructuredOutputParser,ResponseSchema


# Load environment variables
load_dotenv()

model=GoogleGenerativeAI(model='gemini-1.5-pro')


schema=[
    ResponseSchema(name='fact_1',description='fact 1 about the topic'),
    ResponseSchema(name='fact_2',description='fact 2 about the topic'),
    ResponseSchema(name='fact_3',description='fact 3 about the topic')
]
parser=StructuredOutputParser.from_response_schemas(schema)
template1 = PromptTemplate(
   template="Give me 3 facts about {topic}.\n{format_instruction}",
    input_variables=['topic'],
    partial_variables={"format_instruction": parser.get_format_instructions()}
)

prompt=template1.invoke({'topic':'book reading'})

result=model.invoke(prompt)
final_result=parser.parse(result)
print(final_result)