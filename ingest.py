import pinecone
import os
import json

from langchain.vectorstores import Pinecone
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import TokenTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# LOCAL STUFF
# Set Key Variables
#from dotenv import load_dotenv
#load_dotenv()
#pinecone.api_key = os.getenv("PINECONE_API_KEY")
root = "/Users/christienkerbert/Desktop/"
with open(f'{root}api_keys.json', 'r') as inp:
  keys = json.load(inp)

openai.organization = "org-8oWTVWLA0ES5yhsucMCuX5c0"
openai.api_key = keys['openai']['api_key']
os.environ['OPENAI_API_KEY'] = keys['openai']['api_key']
# END LOCAL STUFF

"""
Pinecone Setup
"""
# Initialize pinecone
#pinecone.init(
#    environment="us-central1-gcp"  # next to api key in console
#)
#index_name = "support-kb"
"""
END Pinecone Setup
"""

# First, we need to initialize load URLs and load the text
# Opens the file and reads the URLs
#with open('./urls.txt', 'r') as file:
        #urls = [line.strip() for line in file]
pdf_path = "/whirlpool-dishwasher.pdf"

# Loads the list of URLS and all the text (Consider using Selenium URL loader)
#loader = UnstructuredURLLoader(urls=urls)
#kb_data = loader.load()

# Second, split the text into chunks, using Tiktoken
#text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=100)
#kb_docs = text_splitter.split_documents(kb_data) #Note that split_documents works but not split_text because we're dealing with a list of documents

# Create the embeddings
embeddings = OpenAIEmbeddings()

# Store in the DB
kb_db_store = Pinecone.from_documents(kb_docs, embeddings, index_name=index_name, namespace="lsvt")