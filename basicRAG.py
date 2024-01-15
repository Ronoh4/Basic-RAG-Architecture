#Import necessary modules and correct classes
from llama_index import download_loader
from dotenv import load_dotenv
from llama_index.node_parser import SimpleNodeParser
from llama_index.vector_stores import DeepLakeVectorStore
from llama_index import GPTVectorStoreIndex
from llama_index.storage.storage_context import StorageContext
from llama_index.embeddings import OpenAIEmbedding
from llama_index import ServiceContext
import os

#Loading env variables from .env file named 'llamawiki' and get key values
load_dotenv('llamawiki.env')
activeloop_token = os.getenv('ACTIVELOOP_TOKEN')
openai_key = os.getenv('OPENAI_API_KEY')
activeloop_org_id = os.getenv('ACTIVELOOP_ORG_ID')
activeloop_dataset_name = os.getenv('ACTIVELOOP_DATASET_NAME')

#Download Wikipedia loader and load data from 3 Wikipedia pages specified
WikipediaReader = download_loader("WikipediaReader")
loader = WikipediaReader()
documents = loader.load_data(pages=['Prompt engineering', 'Database', 'Embedding'])
print(len(documents))

#Create a parser, and parse and chunk the documents into nodes 
parser = SimpleNodeParser.from_defaults(chunk_size=400, chunk_overlap=40)
nodes = parser.get_nodes_from_documents(documents)
print(len(nodes))

#Define dataset path and creating a DeepLake vectorstore with the defined path
dataset_path = f"hub://{activeloop_org_id}/{activeloop_dataset_name}"
vector_store = DeepLakeVectorStore(dataset_path=dataset_path)

#Create an embedding model and ServiceContext as default (to manage services (OpenAIEmbedding) and storage)
embed_model = OpenAIEmbedding(api_key=openai_key)
service_context = ServiceContext.from_defaults(embed_model=embed_model)

#Try to create an index and catch any errors
try:
    index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context, vector_store=vector_store)
except Exception as e:
    print(f"Error in indexing: {e}")

#Try to query the index and catch any errors
try:
    query_engine = index.as_query_engine()
    response = query_engine.query("What does Embedding mean?")
    print(response.response)
except Exception as e:
    print(f"Error quering index: {e}")