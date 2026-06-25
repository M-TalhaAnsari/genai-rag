from langchain_google_genai import ChatGoogleGenerativeAI
from pypdf import PdfReader
import pandas as pd
import os
import json
from langchain_classic.prompts import PromptTemplate
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.llm import LLMChain
from dotenv import load_dotenv
load_dotenv()

from langsmith import traceable, Client
client = Client()
api_key = os.getenv("GOOGLE_API_KEY")
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
langsmith_tracing = os.getenv("LANGCHAIN_TRACING_V2")
os.environ["GOOGLE_API_KEY"] = str(api_key)
os.environ["LANGCHAIN_API_KEY"] = str(langchain_api_key)
os.environ["LANGCHAIN_TRACING_V2"] = str(langsmith_tracing)


llm = ChatGoogleGenerativeAI(api_key = os.getenv("GOOGLE_API_KEY"),model="gemini-2.5-flash", temperature =0, max_tokens=2000)

from kor.extraction import create_extraction_chain
from kor.nodes import Object, Text, Number

# iterate over files that user uploaded pdf files, one by one
@traceable(
    run_type="chain",
    name = "OpenAI-Call Decorator 1",
    tags = ["samplechain"],
    metadata = {"chainname": "simplechain"}
)
def createdocs(user_pdf_file):
    """
    This function is to extract the invoice the data from the given pdf file
    it uses the langchain agent to extract the data from the given pdf file
    """
    df = pd.DataFrame({
        'Invoice no.': pd.Series(dtype='str'),
        'Description':pd.Series(dtype='str'),
        'Quantity': pd.Series(dtype='str'),
        'Date': pd.Series(dtype='str'),
        'Unit_price': pd.Series(dtype='str'),
        'Amount': pd.Series(dtype='str'),
        'Total': pd.Series(dtype='str'),
        'Email': pd.Series(dtype='str'),
        'Phone number': pd.Series(dtype='str'),
        'Address': pd.Series(dtype='str'),
    })

    for filename in user_pdf_file:
        text =""
        pdf_reader = PdfReader(filename)
        for page in pdf_reader.pages:
            text += page.extract_text()

        template = """Extract all the following values : invoice no., Description, Quantity, date, 
            Unit price , Amount, Total, email, phone number and address from the following Invoice content: 
            {texts}
            The fields and values in the above content may be jumbled up as they are extracted from a PDF. Please use your judgement to align
            the fields and values correctly based on the fields asked for in the question abiove.
            Expected output format: 
            {{'Invoice no.': xxxxxxxx','Description': 'xxxxxx','Quantity': 'x','Date': 'dd/mm/yyyy',
            'Unit price': xxx.xx','Amount': 'xxx.xx,'Total': xxx,xx,'Email': 'xxx@xxx.xxx','Phone number': 'xxxxxxxxxx','Address': 'xxxxxxxxx'}}
            Remove any dollar symbols or currency symbols from the extracted values.
            """
        prompt = PromptTemplate(input_variables=["texts"], template=template)

        llm = ChatGoogleGenerativeAI(
            model='gemini-2.5-flash', 
            temperature=0, 
            model_kwargs={"response_mime_type": "application/json"}
        )
        chain = LLMChain(llm = llm, prompt = prompt)

        data_dict = chain.run(text)

        print(data_dict)

        parsed_data = json.loads(data_dict)

        new_row_pdf = pd.DataFrame([json.loads(data_dict)])
        new_row_pdf = new_row_pdf.reindex(columns=df.columns)

        df = pd.concat([df, new_row_pdf], ignore_index = True)

    print(df)
    return df