#import pinecone
import promptlayer
import os
import json
import gradio as gr
import re
import openai

#from langchain.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import PromptLayerChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader


# LOCAL STUFF
# Set Key Variables
#from dotenv import load_dotenv
#load_dotenv()
#promptlayer.api_key = os.getenv("PROMPTLAYER_API_KEY")
#pinecone.api_key = os.getenv("PINECONE_API_KEY")
root = "/Users/christienkerbert/Desktop/"
with open(f'{root}api_keys.json', 'r') as inp:
  keys = json.load(inp)

openai.organization = "org-8oWTVWLA0ES5yhsucMCuX5c0"
openai.api_key = keys['openai']['api_key']
os.environ['OPENAI_API_KEY'] = keys['openai']['api_key']
promptlayer.api_key = keys['promptlayer']['api_key']
os.environ['PROMPTLAYER_API_KEY'] = keys['promptlayer']['api_key']
# END LOCAL STUFF

pdf_path = "./whirlpool-dishwasher.pdf"

# Loads the list of URLS and all the text (Consider using Selenium URL loader)
loader = PyPDFLoader(pdf_path)
pages = loader.load_and_split()

# Create the embeddings
embeddings = OpenAIEmbeddings()

# Store in the DB
vectordb = Chroma.from_documents(pages, embedding=embeddings,
                                 persist_directory=".")  # Load the pdf into the Chroma database
vectordb.persist()

# Prompt Template & Messages
_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

template = """You are an AI maintenance intake chatbot called Max. 
Max coaches residents through resolving their issues with appliances based on the symptom it receives from the resident and context from appliances' manuals.
When providing an answer, choose a casual but helpful tone of voice. 
Max ensures the resident is always safe and doesn't perform any actions that require specialized knowledge, such as touching electrical wiring.
Assume the resident doesn't own replacement parts except for light bulbs.
Don't ask the resident to contact a professional themselves.
End the conversation by asking if the proposed solution worked.
Use html bullet list format when needed.
Question: {question}
=========
{context}
=========
"""
QA_PROMPT = PromptTemplate(template=template, input_variables=["question", "context"])

# Initialize Pinecone
#pinecone.init(
#    environment="us-central1-gcp"
#)
#index_name = "support-kb"

# Replace kb_db_store initialization with Pinecone.from_existing_index method
#embeddings = OpenAIEmbeddings()
#kb_db = Pinecone.from_existing_index(index_name=index_name, embedding=embeddings, namespace="lsvt")


# Third, we need to create the prompt
# Initialize the ChatVectorDBChain
kb_chat = ConversationalRetrievalChain.from_llm(
    PromptLayerChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", pl_tags=["local", "customTemp7"]),
    retriever=vectordb.as_retriever(),
    verbose=True, 
    return_source_documents=True,
    combine_docs_chain_kwargs={'prompt': QA_PROMPT},
    condense_question_prompt=CONDENSE_QUESTION_PROMPT,
)

# Format the text in the return
def format_terms(text):
    formatted_text = re.sub(r'[“"”]([^”“]+)[“"”]', r'<b>\1</b>', text)
    formatted_text = formatted_text.replace('\n', '<br>')
    return formatted_text


chat_history = []
def get_answer(query):
    
    result = kb_chat({"question": query, "chat_history": chat_history})
    chat_history.append((query, result["answer"])) 

     # Use a set to remove duplicates
    source_urls = list(set(doc.metadata["source"] for doc in result["source_documents"]))  
    
    formatted_answer = format_terms(result['answer'])    
    
    answer = f"<strong>Answer:</strong><br><br> {formatted_answer}<br><br><strong>Source URLs:</strong><br>"
    answer += "<br>".join(f'<a href="{url}" target="_blank">{url}</a>' for url in source_urls)
    
    return answer

# Gradio Interface
def user(user_message, history):
    return "", history + [[user_message, None]]

def bot(history):
    query = history[-1][0]
    result = get_answer(query)
    history[-1][1] = result
    return history

# Launch the Gradio interface with the chatbot components and functions
with gr.Blocks() as demo:
    chatbot = gr.Chatbot()

    with gr.Row():
        msg = gr.Textbox(
            label="What's your question?",
            placeholder="What's the answer to life, the universe, and everything?",
            lines=2,
        )
        submit = gr.Button(value="Submit", variant="primary").style(full_width=False)
        clear = gr.Button("Clear").style(full_width=False)

    gr.Examples(
        examples = [
            ["How do I create a new user?"],
            ["what's new in the past month?"],
            ["what was released in October?"],
            ["how do I add files to the media library?"],
            ["what is a postable?"],
            ["what formats and resolutions do you support in the media library?"],
            ["how do I configure a certification?"]
        ],
        inputs=msg,
    )

    submit.click(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, chatbot
    )
    clear.click(lambda: None, None, chatbot, queue=False)

    with gr.Row():
        msg
        submit
        clear

demo.launch(share=True)
#get_answer("My dishwasher is not draining. What can I do?")