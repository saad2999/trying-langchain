from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser,PydanticOutputParser
from pydantic import BaseModel,Field
from typing import Literal
from langchain.schema.runnable import RunnableBranch,RunnableLambda


load_dotenv()
model1=GoogleGenerativeAI(model='gemini-1.5-pro')

class Feedback(BaseModel):
    sentiment:Literal['positive','negative']=Field(description="sentiment value in positive or negative")

parser2=PydanticOutputParser(pydantic_object=Feedback)

prompt1=PromptTemplate(
    template="classify the sentiment of following feedback into negitive or positive\n {feedback}\n{format_instruction}",
    input_variables=['feedback'],
    partial_variables={'format_instruction':parser2.get_format_instructions()}
)
prompt2 = PromptTemplate(
    template="Write an appropriate response to this positive feedback:\n{feedback}",
    input_variables=["feedback"]
)
prompt3 = PromptTemplate(
    template="Write an appropriate response to this negative feedback:\n{feedback}",
    input_variables=["feedback"]
)

parser=StrOutputParser()
classlifier_chain=prompt1|model1|parser2
branch_chain=RunnableBranch(
    (lambda x:x.sentiment=='positive',prompt2|model1|parser),
    (lambda x:x.sentiment=='negative',prompt3|model1|parser),
    RunnableLambda(lambda x: 'could not find sentiment')
    
)
chain=classlifier_chain|branch_chain
r=chain.invoke({'feedback':'this is a good phone'})
print(r)
chain.get_graph().print_ascii()