import os
from typing import Dict, List
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# 1. Define the strict structure the model must return
class VerificationSchema(BaseModel):
    supported: str = Field(description="Is the answer supported by the context? Output exactly 'YES' or 'NO'.")
    unsupported_claims: List[str] = Field(description="List of any unsupported claims made in the answer. Empty list if none.")
    contradictions: List[str] = Field(description="List of any contradictions against the context. Empty list if none.")
    relevant: str = Field(description="Is the answer relevant to the context? Output exactly 'YES' or 'NO'.")
    additional_details: str = Field(description="Any additional explanations, details, or rationale.")


class VerificationAgent:
    def __init__(self) -> None:
        """Initialize the verification agent with the model and structured output schema."""
        # Initialize the base Gemini model (temperature=0 makes it more factual and deterministic)
        base_model = ChatGoogleGenerativeAI(
            api_key=os.environ.get("GOOGLE_API_KEY"), 
            model="gemini-2.5-flash",
            temperature=0.0 
        )
        # Bind the Pydantic schema to the model
        self.model = base_model.with_structured_output(VerificationSchema)

    def generate_prompt(self, answer: str, context: str) -> str:
        """Generate a prompt for the LLM to verify the answer against the context."""
        return f"""
        You are an AI assistant designed to verify the accuracy and relevance of answers based on the provided context.
        
        Instructions:
        1. Verify the provided Answer against the provided Context.
        2. Extract any unsupported claims or contradictions.
        3. Fill out the requested fields accurately based on the context.
        
        Answer: {answer}
        
        Context:
        {context}
        """

    def format_verification_report(self, verification: VerificationSchema) -> str:
        """Format the structured Pydantic object into a readable paragraph for the UI."""
        report = f"Supported: {verification.supported}\n"
        
        if verification.unsupported_claims:
            report += f"Unsupported Claims: {', '.join(verification.unsupported_claims)}\n"
        else:
            report += "Unsupported Claims: None\n"
            
        if verification.contradictions:
            report += f"Contradictions: {', '.join(verification.contradictions)}\n"
        else:
            report += "Contradictions: None\n"
            
        report += f"Relevant: {verification.relevant}\n"
        
        if verification.additional_details:
            report += f"Additional Details: {verification.additional_details}\n"
        else:
            report += "Additional Details: None\n"
            
        return report

    def check(self, answer: str, documents: List[Document]) -> Dict:
        """
        Verify the answer against the provided documents.
        This method maintains your original input/output signature so it won't break your app.
        """
        print(f"VerificationAgent.check called with {len(documents)} documents.")
        
        # Combine all document contents into one string
        context = "\n\n".join([doc.page_content for doc in documents])
        prompt = self.generate_prompt(answer, context)
        
        print("Sending prompt to the model (awaiting structured output)...")
        try:
            # invoke() directly returns a VerificationSchema object, no string parsing needed!
            verification_result: VerificationSchema = self.model.invoke(prompt)
            print("LLM response successfully mapped to Pydantic schema.")
            
        except Exception as e:
            print(f"Error during model inference or parsing: {e}")
            # Safe fallback if the API fails, ensuring the program keeps running
            verification_result = VerificationSchema(
                supported="NO",
                unsupported_claims=[],
                contradictions=[],
                relevant="NO",
                additional_details=f"Model verification failed due to error: {str(e)}"
            )

        # Convert the object into the formatted string your UI expects
        verification_report_formatted = self.format_verification_report(verification_result)
        print(f"Verification report:\n{verification_report_formatted}")

        # Return exactly the format your workflow expects
        return {
            "verification_report": verification_report_formatted,
            "context_used": context
        }