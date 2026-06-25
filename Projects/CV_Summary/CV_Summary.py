
import docx2txt
import pypdf
from dotenv import load_dotenv
from langchain_community.document_loaders import Docx2txtLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
import streamlit as st
import textwrap
import os
load_dotenv()



model = ChatGoogleGenerativeAI(model='gemini-2.5-flash', api_key="YOUR API KEY", temperature=0.5)




def process_docx(uploaded_file):
    # Read DOCX directly from browser stream memory
    text = docx2txt.process(uploaded_file)
    
    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=1000,
        chunk_overlap=50
    )
    # Return as a list of LangChain Document objects
    return text_splitter.create_documents([text])


def process_pdf(uploaded_file):
    # Read PDF directly from browser stream memory
    pdf_reader = pypdf.PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
        
    text = text.replace('\t', ' ')

    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=1000,
        chunk_overlap=50
    )
    return text_splitter.create_documents([text])




def main(model):
    st.title("CV Summary Generator")
    uploaded_file = st.file_uploader("Select CV", type=['docx', 'pdf'])

    text = ""

    if uploaded_file is not None:
        file_extension = uploaded_file.name.split('.')[-1]

        st.write("File Details")
        st.write(f"File Name: {uploaded_file.name}")
        st.write(f"File Type: {file_extension}")

        if file_extension == 'docx':
            text = process_docx(uploaded_file)
        elif file_extension == 'pdf':
            text = process_pdf(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload a .docx or .pdf file")

            return 

        prompt_template = """
            You have been given a resume to analyze. Write a verbose detail of the following:
            {text}
            Details:
        """
        prompt = PromptTemplate.from_template(prompt_template)

        refine_template = (

            "Your job is to produce a final outcome\n"
            "We have provided an existing detail: {existing_answer}\n"
            "We want a refined version of the existing detail based on initial details below\n"
            "------------\n"
            "{text}\n"
            "------------\n"
            "Given the new context, refine the original summary in the following manner:"
            "Name: \n"
            "Email: \n"
            "Key Skills: \n"
            "Last Company: \n"
            "Experience Summary: \n"

        )

        refine_prompt = PromptTemplate.from_template(refine_template)

        chain = load_summarize_chain(
            llm = model,
            chain_type = "refine",
            question_prompt = prompt,
            refine_prompt = refine_prompt,
            return_intermediate_steps = True,
            input_key = "input_documents",
            output_key = "output_text",
        )
        result = chain({"input_documents": text}, return_only_outputs=True)

        st.write("Resume Summary:")
        st.text_area("Text", result['output_text'], height = 400)


if __name__ == "__main__":
    main(model)

