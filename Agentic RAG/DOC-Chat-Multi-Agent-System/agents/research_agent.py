import os
import json
from typing import TypedDict, List, Dict
from dotenv import load_dotenv

# Keeping your project's specific imports intact
from langchain_classic.schema import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from config.settings import settings

load_dotenv()

class ResearchAgent:
    def __init__(self) -> None:
        """
        Initialize the research agent with the Gemini model.
        """
        self.model = ChatGoogleGenerativeAI(
            api_key=os.environ.get("GOOGLE_API_KEY"), 
            model="gemini-2.5-flash",
            temperature=0.2  # Slightly lower temperature to encourage tight, factual adherence to context
        )
    
    def sanitize_response(self, response_text: str) -> str:
        """Sanitize the llm response by stripping unnecessary whitespace."""
        return response_text.strip()
    
    def generate_prompt(self, question: str, context: str) -> str:
        """
        Generate a structured prompt for the LLM to generate a precise and factual answer.
        """
        prompt = f"""
        You are an AI assistant designed to provide precise and factual answers based on the given context.
        Instructions:
        - Answer the following question using only the provided context.
        - Be clear, concise, and factual.
        - Return as much information as you can get from the context.
        Question: {question}
        Context:
        {context}
        Provide your answer below:
        """
        return prompt
    
    def generate(self, question: str, documents: List[Document]) -> Dict:
        """
        Generate an initial answer using the provided documents.
        """
        print(f"ResearchAgent.generate called with question='{question}' and {len(documents)} documents.")
        
        # Combine the top document contents into one string
        context = "\n\n".join([doc.page_content for doc in documents])
        print(f"Combined context length: {len(context)} characters.")

        # Create the prompt for the llm
        prompt = self.generate_prompt(question, context)
        
        print("Sending prompt to the Gemini model...")
        try:
            # FIX 1: Use LangChain's native .invoke() for passing strings to ChatGoogleGenerativeAI
            response = self.model.invoke(prompt)
            print("LLM response received.")
            
            # FIX 2: LangChain outputs an AIMessage object. Access the text using the .content attribute
            llm_response = response.content.strip()
            print(f"Raw LLM response:\n({llm_response})")
            
        except Exception as e:
            print(f"Error during model inference or processing: {e}")
            # Fallback error string instead of crashing the entire LangGraph workflow pipeline execution
            llm_response = "I cannot answer this question based on the provided documents due to a processing error."
        
        # FIX 3: Defined the missing sanitize_response method above so this line works seamlessly
        draft_answer = self.sanitize_response(llm_response) if llm_response else "I cannot answer this question based on the provided documents."
        print(f"Generated answer: {draft_answer}")

        return {
            "draft_answer": draft_answer,
            "context_used": context
        }