import streamlit as st 
from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import load_prompt

load_dotenv()
st.header("Research Tool")
paper_input = st.selectbox(
    "Select Research Paper Name",
    [
        "Attention Is All You Need",
        "BERT: Pre-training of Deep Bidirectional Transformers",
        "GPT-3: Language Models are Few-Shot Learners",
        "Diffusion Models Beat GANs on Image Synthesis"
    ]
)
style_input = st.selectbox(
    "Select Explanation Style",
    ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"]
)

length_input = st.selectbox(
    "Select Explanation Length",
    ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"]
)


chatmodel=GoogleGenerativeAI(model='gemini-1.5-pro')
template=load_prompt("pt.json")

prompt=template.invoke({
    "paper_input": paper_input,
    "style_input": style_input,
    "length_input": length_input,
})
result=None
if st.button("Summarize"):
    result=chatmodel.invoke(prompt)
    st.write(result)
print(result)