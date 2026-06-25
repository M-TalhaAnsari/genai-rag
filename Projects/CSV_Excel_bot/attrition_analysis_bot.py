import pandas as pd
from dotenv import load_dotenv
from langchain_classic.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st
import os

def main():
    load_dotenv()
    df = pd.read_csv("HR-Employee-Attrition.csv")

    st.set_page_config(
        page_title="Documentation Chatbot",
        page_icon = ":books",
    )

    st.title("Attribution Analysis Chatbot")
    st.subheader("Uncover insight from attrition Data!")

    st.markdown(
        """
        This chatbot is created to answer questions from a set of attrition data from your organization.
        Ask a question and the chatbot will respond with appropriete Analysis
            """
    )
    st.write(df.head())
    user_question = st.text_input("Ask your question about the data")

    llm_model = ChatGoogleGenerativeAI(
            model='gemini-2.5-flash',
            temperature=0
        )
    agent = create_csv_agent(
        llm_model,
        "HR-Employee-Attrition.csv",
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        allow_dangerous_code=True
    )
    
    if user_question:
        with st.spinner("Analyzing data..."):
            # 🟢 FIXED: Changed legacy agent.run() to the modern agent.invoke()
            response = agent.invoke({"input": user_question})
            
            st.write("### Analysis Result:")
            # Agent executors return a dictionary; the final text is under the 'output' key
            st.write(response["output"])

if __name__=="__main__":
    main()