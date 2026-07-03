import os
import json
from typing import List, Optional
import asyncio
import warnings
import numpy as np
warnings.filterwarnings('ignore')

# Core LlamaIndex imports
from llama_index.core import (
    VectorStoreIndex, 
    SimpleDirectoryReader, 
    Document,
    Settings,
    DocumentSummaryIndex,
    KeywordTableIndex
)
from llama_index.core.retrievers import (
    BaseRetriever,
    VectorIndexRetriever,
    AutoMergingRetriever,
    RecursiveRetriever,
    QueryFusionRetriever
)
from llama_index.core.indices.document_summary import (
    DocumentSummaryIndexLLMRetriever,
    DocumentSummaryIndexEmbeddingRetriever,
)
from llama_index.core.node_parser import SentenceSplitter, HierarchicalNodeParser
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.embeddings import BaseEmbedding
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from llama_index.llms.google_genai import GoogleGenAI
# Advanced retriever imports
from llama_index.retrievers.bm25 import BM25Retriever

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("⚠️ scipy not available - some advanced fusion features will be limited")

print("✅ All imports successful!")
load_dotenv()

llm = GoogleGenAI(
    model="gemini-2.5-flash",
    api_key=os.getenv("GOOGLE_API_KEY"),
)

import os
from dotenv import load_dotenv
from llama_index.llms.google_genai import GoogleGenAI

load_dotenv()

def create_google_llm():
    """Create Google Gemini LLM instance using the official LlamaIndex integration."""
    try:
        llm = GoogleGenAI(
            model="gemini-2.5-flash",  # or "gemini-2.5-pro"
            api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.9,
        )

        print("✅ Google Gemini LLM initialized successfully")
        return llm

    except Exception as e:
        print(f"⚠️ Google Gemini initialization error: {e}")
        print("Falling back to MockLLM for demonstration")

        from llama_index.core.llms.mock import MockLLM
        return MockLLM(max_tokens=512)
    
print(" Initializing HuggingFace embeddings...")
embed_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)
llm = create_google_llm()

Settings.llm = llm
Settings.embed_model = embed_model

# Sample data for the lab - AI/ML focused documents
SAMPLE_DOCUMENTS = [
    "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.",
    "Deep learning uses neural networks with multiple layers to model and understand complex patterns in data.",
    "Natural language processing enables computers to understand, interpret, and generate human language.",
    "Computer vision allows machines to interpret and understand visual information from the world.",
    "Reinforcement learning is a type of machine learning where agents learn to make decisions through rewards and penalties.",
    "Supervised learning uses labeled training data to learn a mapping from inputs to outputs.",
    "Unsupervised learning finds hidden patterns in data without labeled examples.",
    "Transfer learning leverages knowledge from pre-trained models to improve performance on new tasks.",
    "Generative AI can create new content including text, images, code, and more.",
    "Large language models are trained on vast amounts of text data to understand and generate human-like text."
]

# Consistent query examples used throughout the lab
DEMO_QUERIES = {
    "basic": "What is machine learning?",
    "technical": "neural networks deep learning", 
    "learning_types": "different types of learning",
    "advanced": "How do neural networks work in deep learning?",
    "applications": "What are the applications of AI?",
    "comprehensive": "What are the main approaches to machine learning?",
    "specific": "supervised learning techniques"
}


class AdvancedRetrievers:
    def __init__(self):

        self.documents = [Document(text=text) for text in SAMPLE_DOCUMENTS]
        self.nodes = SentenceSplitter().get_nodes_from_documents(self.documents)

        self.vector_index = VectorStoreIndex.from_documents(self.documents)
        self.document_summary_index = DocumentSummaryIndex.from_documents(self.documents)
        self.keyword_index = KeywordTableIndex.from_documents(self.documents)
        
        print(f"Loaded {len(self.documents)} documents")
        print(f"Created {len(self.nodes)} nodes")


indexes = AdvancedRetrievers()


# VECTOR INDEX RETRIEVER
vector_retriever = VectorIndexRetriever(
    index = indexes.vector_index,
    similarity_top_k = 3
)

# Alternative creation method
all_retrievers = indexes.vector_index.as_retriever(similarity_top_k=3)

query = DEMO_QUERIES["basic"]
nodes = vector_retriever.retrieve(query)

# BM25 RETRIEVE - ADVANCED KEYWORD-BASED SEARCH
# Improved TF-IDF limitation. Improvements:
# 1. Term Frequency Saturation
# 2. Document length Normalization
# 3. Tunable Parameters

try:
    import Stemmer

    bm25_retrieve = BM25Retriever.from_defaults(
        nodes=indexes.nodes,
        similarity_top_k=3,
        stemmer = Stemmer.Stemmer("english"),
        language="english"
    )
    query = DEMO_QUERIES["technical"]
    nodes = bm25_retrieve.retrieve(query)

except ImportError:
    fallback_retriever = indexes.vector_index.as_retriever(similarity_top_k=3)
    query = DEMO_QUERIES["technical"]
    nodes = fallback_retriever.retrieve(query)


# DOCUMENT SUMMARY INDEX RETRIEVE
# First uses summary to filters documents, then return full document content
# Two Option
# DocumentSummaryIndesLLMRetriever
# DocumentSummaryIndexEmbeddingRetriver
doc_summary_retriever_llm = DocumentSummaryIndexLLMRetriever(
    indexes.document_summary_index,
    choice_top_k=3
)

doc_summary_retriever_embedding = DocumentSummaryIndexEmbeddingRetriever(
    indexes.document_summary_index,
    similarity_top_k = 3
)
query = DEMO_QUERIES["learning_types"]

try:
    nodes_llm = doc_summary_retriever_llm.retrieve(query)
    print(f"Recieved {len(nodes_llm)} nodes")

except Exception as e:
    print(f"LLM-based retrieval demo: {str(e)[:100]}...")

try:
    nodes_emb = doc_summary_retriever_embedding.retrieve(query)
    print(f"Retrieved {len(nodes_emb)} nodes")
except Exception as e:
    print(f"Embedding-based retrieval demo: {str(e)[:100]}...")

# AUTO-MERGING RETRIEVER
# if enough child nodes from the same parent retrieved, the retriever returns the parent node instead

nodes_parser = HierarchicalNodeParser.from_defaults(
    chunk_sizes = [512, 256, 128]
)
hier_nodes = nodes_parser.get_nodes_from_documents(indexes.documents)

from llama_index.core import StorageContext
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.vector_stores import SimpleVectorStore

docstore = SimpleDocumentStore()
docstore.add_documents(hier_nodes)

storage_context = StorageContext.from_defaults(docstore=docstore)

base_index = VectorStoreIndex(hier_nodes, storage_context=storage_context)
base_retriever = base_index.as_retriever(similarity_top_k=6)

auto_merging_retriever = AutoMergingRetriever(
    base_retriever,
    storage_context,
    verbose = True
) 
query = DEMO_QUERIES["advanced"]
nodes = auto_merging_retriever.retrieve(query)

# RECURSIVE RETRIVER
# designed to follow relationships between nodes using reference
# used for research paper and matedata links

# Create documents with references
docs_with_refs = []
for i, doc in enumerate(lab.documents):
    # Add reference metadata
    ref_doc = Document(
        text=doc.text,
        metadata={
            "doc_id": f"doc_{i}",
            "references": [f"doc_{j}" for j in range(len(lab.documents)) if j != i][:2]
        }
    )
    docs_with_refs.append(ref_doc)

ref_index = VectorStoreIndex.from_documents(docs_with_refs)

retriever_dict = {
    f"doc_{i}": ref_index.as_retriever(similarity_top_k=1)
    for i in range(len(docs_with_refs))
}

# base retriever
base_retriever = ref_index.as_retriever(similarity_top_k=2)
retriever_dict['vector'] = base_retriever

recursive_retriever = RecursiveRetriever(
    "vector",
    retriever_dict=retriever_dict,
    query_engine_dict={},
    verbose=True
)

query = DEMO_QUERIES["applications"]
nodes = recursive_retriever.retrieve(query)
print(f"Recursively retrieved {len(nodes)} nodes")


# QUERY FUSION RETRIEVER - MULTI QUERY ENHANCEMENT WITH ADVANCED FUSION
# Combine rsult from different retrieve and optionally generates multiple variation of a query using llm to imporve coverage
# Create base retriever
base_retriever = indexes.vector_index.as_retriever(similarity_top_k=3)

query = DEMO_QUERIES["comprehensive"]  # "What are the main approaches to machine learning?"
print(f"Query: {query}")
print("QueryFusionRetriever generates multiple query variations and fuses results")
print("using one of three sophisticated fusion modes.")

print("\nOverview of Fusion Modes:")
print("1. RECIPROCAL_RERANK: Uses reciprocal rank fusion (most robust)")
print("2. RELATIVE_SCORE: Preserves score magnitudes (most interpretable)")  
print("3. DIST_BASED_SCORE: Statistical normalization (most sophisticated)")

print("\nDemonstration workflow:")
print("Each subsection below explores one fusion mode in detail with:")
print("- Theoretical explanation of the fusion method")
print("- Live demonstration using QueryFusionRetriever")
print("- Manual implementation showing the underlying mathematics")
print("- Use case recommendations and trade-offs")

print(f"\nUsing consistent test query throughout: '{query}'")
print("This allows direct comparison of how each fusion mode handles the same input.")

print("\nProceed to subsections 6.1, 6.2, and 6.3 for detailed demonstrations...")

# Reciprocal Rank Fusion
base_retriever = indexes.vector_index.as_retriever(similarity_top_k=5)
query = DEMO_QUERIES["comprehensive"] 
try:
    rrf_query_fusion = QueryFusionRetriever(
            [base_retriever],
            similarity_top_k=3,
            num_queries=3,
            mode="reciprocal_rerank",
            use_async=False,
            verbose=True
        )
    nodes = rrf_query_fusion.retrieve(query)
except Exception as e:
    print("Demonstrating RRF concept manually with query variations...")
    query_variations = [
        DEMO_QUERIES["comprehensive"],  # Original query
        "machine learning approaches and methods",
        "different ML techniques and algorithms"
    ]
    all_results = {}
    
    for i, query_var in enumerate(query_variations):
        print(f"\nQuery variation {i+1}: {query_var}")
        nodes = base_retriever.retrieve(query_var)
        
        # Apply RRF scoring
        for rank, node in enumerate(nodes):
            node_id = node.node.node_id
            if node_id not in all_results:
                all_results[node_id] = {
                    'node': node,
                    'rrf_score': 0,
                    'query_ranks': []
                }
    k = 60  # Standard RRF parameter
    rrf_contribution = 1.0 / (rank + 1 + k)
    all_results[node_id]['rrf_score'] += rrf_contribution
    all_results[node_id]['query_ranks'].append((i, rank + 1))
    
    # Sort by final RRF score
    sorted_results = sorted(
        all_results.values(), 
        key=lambda x: x['rrf_score'], 
        reverse=True
    )
    
    print(f"\nCombined RRF Results (top 3):")
    for i, result in enumerate(sorted_results[:3], 1):
        print(f"{i}. Final RRF Score: {result['rrf_score']:.4f}")
        print(f"   Query ranks: {result['query_ranks']}")
        print(f"   Text: {result['node'].text[:100]}...")

# Relative Score Fusion Mode
# normalize retrieval score relative to the maximum score within each query 
base_retriever = indexes.vector_index.as_retriever(similarity_top_k=5)
query = DEMO_QUERIES["comprehensive"]  # "What are the main approaches to machine learning?"

try:
    # Create query fusion retriever with relative score mode
    rel_score_fusion = QueryFusionRetriever(
        [base_retriever],
        similarity_top_k=3,
        num_queries=3,
        mode="relative_score",
        use_async=False,
        verbose=False
    )
    nodes = rel_score_fusion.retrieve(query)
except Exception as e:
    print("Manual Relative Score Fusion with Query Variations:")
    all_results = {}
    query_max_scores = []
    
    # Step 1: Get results and find max scores for each query
    for i, query_var in enumerate(query_variations):
        print(f"\nQuery variation {i+1}: {query_var}")
        nodes = base_retriever.retrieve(query_var)
        scores = [node.score or 0 for node in nodes]
        max_score = max(scores) if scores else 1.0
        query_max_scores.append(max_score)
        
        print(f"Max score for this query: {max_score:.4f}")
        
        # Store results with normalization info
        for node in nodes:
            node_id = node.node.node_id
            original_score = node.score or 0
            normalized_score = original_score / max_score if max_score > 0 else 0
            
            if node_id not in all_results:
                all_results[node_id] = {
                    'node': node,
                    'combined_score': 0,
                    'contributions': []
                }
            
            all_results[node_id]['combined_score'] += normalized_score
            all_results[node_id]['contributions'].append({
                'query': i,
                'original': original_score,
                'normalized': normalized_score
            })
    
    # Step 2: Sort by combined relative score
    sorted_results = sorted(
        all_results.values(),
        key=lambda x: x['combined_score'],
        reverse=True
    )
    
    print(f"\nCombined Relative Score Results (top 3):")
    for i, result in enumerate(sorted_results[:3], 1):
        print(f"{i}. Combined Score: {result['combined_score']:.4f}")
        print(f"   Score breakdown:")
        for contrib in result['contributions']:
            print(f"     Query {contrib['query']}: {contrib['original']:.3f} → {contrib['normalized']:.3f}")
        print(f"   Text: {result['node'].text[:100]}...")

# Distribution Based Score Fusion Mode
# use statistical properties of score distribution rom each query variation to normalize and combine retrival results
base_retriever = indexes.vector_index.as_retriever(similarity_top_k=8)
query = DEMO_QUERIES["comprehensive"]

try:
    # Create query fusion retriever with distribution-based mode
    dist_fusion = QueryFusionRetriever(
        [base_retriever],
        similarity_top_k=3,
        num_queries=3,
        mode="dist_based_score",
        use_async=False,
        verbose=False
    )
    nodes = dist_fusion.retrieve(query)
except Exception as e:
    query_variations = [
        DEMO_QUERIES["comprehensive"],  # Original query
        "machine learning approaches and methods",
        "different ML techniques and algorithms"
    ]
    all_results = {}
    variation_stats = []
    
    # Step 1: Collect results and analyze distributions
    for i, query_var in enumerate(query_variations):
        print(f"\nQuery variation {i+1}: {query_var}")
        nodes = base_retriever.retrieve(query_var)
        scores = [node.score or 0 for node in nodes]
        
        # Calculate distribution statistics
        mean_score = np.mean(scores) if scores else 0
        std_score = np.std(scores) if len(scores) > 1 else 1
        min_score = np.min(scores) if scores else 0
        max_score = np.max(scores) if scores else 1
        
        stats_info = {
            'mean': mean_score,
            'std': std_score,
            'min': min_score,
            'max': max_score,
            'nodes': nodes,
            'scores': scores
        }
        variation_stats.append(stats_info)
        # Apply z-score normalization
        for node, score in zip(nodes, scores):
            node_id = node.node.node_id
            
            # Z-score normalization
            if std_score > 0:
                z_score = (score - mean_score) / std_score
            else:
                z_score = 0
            
            # Convert to [0,1] using sigmoid
            normalized_score = 1 / (1 + np.exp(-z_score))
            
            if node_id not in all_results:
                all_results[node_id] = {
                    'node': node,
                    'combined_score': 0,
                    'contributions': []
                }
            
            all_results[node_id]['combined_score'] += normalized_score
            all_results[node_id]['contributions'].append({
                'query': i,
                'original': score,
                'z_score': z_score,
                'normalized': normalized_score
            })
    # Step 2: Sort by combined distribution-based score
    sorted_results = sorted(
        all_results.values(),
        key=lambda x: x['combined_score'],
        reverse=True
    )
#    Distribution-Based Process:"
# 1. Calculate mean and std for each query variation"
#2. Z-score normalize: z = (score - mean) / std"
#3. Sigmoid transform: normalized = 1 / (1 + exp(-z))"
#4. Sum normalized scores across variations")
#5. Results reflect statistical significance across all query forms"