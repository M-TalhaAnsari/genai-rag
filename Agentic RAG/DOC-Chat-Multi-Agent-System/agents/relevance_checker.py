import os
import re
import logging
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
logger = logging.getLogger(__name__)

class RelevanceChecker:
    def __init__(self) -> None:
        self.model = ChatGoogleGenerativeAI(
            api_key=os.environ.get("GOOGLE_API_KEY"), 
            model="gemini-2.5-flash",
            temperature=0.0  # Crucial for classification tasks to keep behavior deterministic
        )

    def check(self, question: str, retriever, k=3) -> str:
        """
        1. Retrieve the top-k document chunks from the global retriever.
        2. Combine them into a single text string.
        3. Pass that text + question to the LLM for classification.
        Returns: "CAN_ANSWER", "PARTIAL", or "NO_MATCH".
        """
        logger.debug(f"RelevanceChecker.check called with question = '{question}' and k = {k}")
        
        # Retrieve doc chunks from the ensemble retriever
        top_docs = retriever.invoke(question)
        if not top_docs:
            logger.debug("No documents returned from retriever.invoke(). Classifying as NO_MATCH.")
            return "NO_MATCH"
        
        # Combine the top k chunk texts into one string
        document_content = "\n\n".join(doc.page_content for doc in top_docs[:k])
        
        # Create the prompt for llm 
        prompt = f"""
        You are an AI relevance checker between a user's question and provided document content.
        Instructions:
        - Classify how well the document content addresses the user's question.
        - Respond with only one of the following labels: CAN_ANSWER, PARTIAL, NO_MATCH.
        - Do not include any additional text or explanation.
        Labels:
        1) "CAN_ANSWER": The passages contain enough explicit information to fully answer the question.
        2) "PARTIAL": The passages mention or discuss the question's topic but do not provide all the details needed for a complete answer.
        3) "NO_MATCH": The passages do not discuss or mention the question's topic at all.
        Important: If the passages mention or reference the topic or timeframe of the question in any way, even if incomplete, respond with "PARTIAL" instead of "NO_MATCH".
        Question: {question}
        Passages: {document_content}
        Respond ONLY with one of the following labels: CAN_ANSWER, PARTIAL, NO_MATCH
        """

        # Call the llm
        try:
            # FIX 1: Use LangChain's native .invoke() for text generation execution
            response = self.model.invoke(prompt)
            
            # FIX 2: Extract text via the .content attribute rather than the OpenAI choices list dict
            llm_response = response.content.strip().upper()
            logger.debug(f"LLM Response: {llm_response}")
            
        except Exception as e:
            logger.error(f"Error during model inference or response mapping: {e}")
            return "NO_MATCH"
            
        print(f"Checker response: {llm_response}")

        # FIX 3: Robust substring validation to handle edge cases like backticks or quotes
        if "CAN_ANSWER" in llm_response:
            classification = "CAN_ANSWER"
        elif "PARTIAL" in llm_response:
            classification = "PARTIAL"
        elif "NO_MATCH" in llm_response:
            classification = "NO_MATCH"
        else:
            logger.warning(f"LLM did not respond with an explicit label. Raw value: {llm_response}")
            classification = "NO_MATCH"

        logger.debug(f"Classification recognized as '{classification}'")
        return classification