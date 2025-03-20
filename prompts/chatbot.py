from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage , HumanMessage,AIMessage

load_dotenv()

chatmodel=GoogleGenerativeAI(model='gemini-1.5-pro')
chat_history=[
    SystemMessage(content="You are an AI assistant. User will you give you a task. Your goal is to complete the task as faithfully as you can."),
]
while True:
    user_input=input("you:")
    chat_history.append(HumanMessage(content=user_input))
    if user_input.lower()=='quit':
        break
    result=chatmodel.invoke(chat_history)
    chat_history.append(AIMessage(content=result))
    print("AI:",result)
