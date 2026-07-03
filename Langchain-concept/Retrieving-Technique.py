from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_classic.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.stores import InMemoryStore
from langchain_core.documents import Document
from langchain_classic.chains.query_constructor.base import AttributeInfo
from langchain_classic.retrievers.self_query.base import SelfQueryRetriever
from lark import lark
import logging
load_dotenv()

# Supress warning generate by our code
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')



def llm():
    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.5,
        max_output_tokens=256,
    )

    return model

def text_splitter(data, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
        length_function = len,
    )
    chunks = text_splitter.split_documents(data)
    return chunks

def embedding_model():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embeddings

# VECTOR STORE-BACKED RETRIEVER
loader = TextLoader('companypolicies.txt')
txt_data = loader.load()

chunks_txt = text_splitter(txt_data, 200, 20)

vector_db = Chroma.from_documents(chunks_txt, embedding_model())

query = "email policy"
retriever = vector_db.as_retriever()
docs = retriever.invoke(query)
print(docs)

# In retrieving technique we can specify search kwargs like k to limit the retrival results
retriever = vector_db.as_retriever(searc_kwargs={"k": 1})
docs = retriever.invoke(query)
print(docs)


# MMR SEARCH
# In this search we retrieve to the query and minimally similar to the previous document
retirever = vector_db.as_retriever(search_type = "mmr")
docs = retriever.invoke(query)
print(docs)

# SIMILARITY SCORE THRESHOLD RETRIEVER
retriever = vector_db.as_retriever(
    search_type = "similarity_score_threshold", search_kwargs= {"score_threshold":0.4}
)
docs = retriever.invoke(query)
print(docs)


# MULTI QUERY RETRIEVER
# It generate multiple queries from different perspective for a given user input query
# It uses llm to generate multiple queries from different perspective based on the user's input query
loader = PyPDFLoader("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/ioch1wsxkfqgfLLgmd-6Rw/langchain-paper.pdf")
pdf_data = loader.load()

chunks_txt = text_splitter(pdf_data, 500, 20)
ids = vector_db.get()["ids"]
vector_db.delete(ids)
vector_db = Chroma.from_documents(documents=chunks_txt, embedding = embedding_model())
query = "What does paper say about langchain"
retriever = MultiQueryRetriever.from_llm(
    retriever=vector_db.as_retriever(), llm = llm()
)
logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)
docs = retriever.invoke(query)
print(docs)

# SELF QUERY RETIRVER
# given a natural laguage query, the retirver uses a query-constructing LLM chain to generate a structured query. It then applies structured query to its underlying vector store
docs = [
    Document(
        page_content="A bunch of scientists bring back dinosaurs and mayhem breaks loose",
        metadata={"year": 1993, "rating": 7.7, "genre": "science fiction"},
    ),
    Document(
        page_content="Leo DiCaprio gets lost in a dream within a dream within a dream within a ...",
        metadata={"year": 2010, "director": "Christopher Nolan", "rating": 8.2},
    ),
    Document(
        page_content="A psychologist / detective gets lost in a series of dreams within dreams within dreams and Inception reused the idea",
        metadata={"year": 2006, "director": "Satoshi Kon", "rating": 8.6},
    ),
    Document(
        page_content="A bunch of normal-sized women are supremely wholesome and some men pine after them",
        metadata={"year": 2019, "director": "Greta Gerwig", "rating": 8.3},
    ),
    Document(
        page_content="Toys come alive and have a blast doing so",
        metadata={"year": 1995, "genre": "animated"},
    ),
    Document(
        page_content="Three men walk into the Zone, three men walk out of the Zone",
        metadata={
            "year": 1979,
            "director": "Andrei Tarkovsky",
            "genre": "thriller",
            "rating": 9.9,
        },
    ),
]
metadata_field_info = [
    AttributeInfo(
        name="genre",
        description="The genre of the movie. One of ['science fiction', 'comedy', 'drama', 'thriller', 'romance', 'action', 'animated']",
        type="string",
    ),
    AttributeInfo(
        name="year",
        description="The year the movie was released",
        type="integer",
    ),
    AttributeInfo(
        name="director",
        description="The name of the movie director",
        type="string",
    ),
    AttributeInfo(
        name="rating", description="A 1-10 rating for the movie", type="float"
    ),
]
vector_db = Chroma.from_documents(docs, embedding_model())
document_content_description = "Brief summary of a movie."

retriver = SelfQueryRetriever.from_llm(
    llm(),
    vector_db,
    document_content_description,
    metadata_field_info
)
docs = retriever.invoke("I want to watch a movie rated higher than 8.5")
print(docs)

# Parent Document Retirver
parent_splitter = CharacterTextSplitter(chunk_size = 1000, chunk_overlap=20, seperator = '\n')
child_splitter = CharacterTextSplitter(chunk_size = 200, chunk_overlap=20, seperator = '\n')

vector_db = Chroma(
    collection_name="split_parents", embedding_function=embedding_model()
)
store = InMemoryStore()
retriever = ParentDocumentRetriever(
    vectorstore=vector_db,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter
)
retriever.add_documents(txt_data)
print(len(list(store.yield_keys())))
sub_docs = vector_db.similarity_search("smoking policy")
retrieved_docs = retriever.invoke("smoking policy")
print(retrieved_docs[0].page_content)

