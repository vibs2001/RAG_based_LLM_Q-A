from flask import Flask, request, render_template
import torch
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.prompts.prompts import SimpleInputPrompt
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.embeddings.langchain import LangchainEmbedding
import sys

app = Flask(__name__)

if torch.cuda.is_available():
    print(f"CUDA is available. using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available")

# Load the documents
documents = SimpleDirectoryReader("/path/to/your/data").load_data()

# System and query prompts
system_prompt = """
You are a Q&A assistant. Your goal is to answer questions as
accurately as possible based on the instructions and context provided.
"""
query_wrapper_prompt = SimpleInputPrompt("<|USER|>{query_str}<|ASSISTANT|>")

# Llama2 Model Setup
llm = HuggingFaceLLM(
    context_window=4096,
    max_new_tokens=320,
    generate_kwargs={"temperature": 0.1, "do_sample": False},
    system_prompt=system_prompt,
    query_wrapper_prompt=query_wrapper_prompt,
    tokenizer_name="meta-llama/Llama-2-7b-chat-hf",# You might need to request access in hugging face
    model_name="meta-llama/Llama-2-7b-chat-hf",# You might need to request access in hugging face
    device_map="auto",
    model_kwargs={"torch_dtype": torch.float16, "load_in_8bit": True}
)

# Embedding Model Setup
embed_model = LangchainEmbedding(HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"))

# Update recursion limit for larger input size
sys.setrecursionlimit(3000)

# Create the index using LlamaIndex without explicitly calling Settings
index = VectorStoreIndex.from_documents(
    documents,
    llm=llm,
    embed_model=embed_model,
    chunk_size=300  # Settings can be set directly here
)

# Define the query route
@app.route('/', methods=['GET', 'POST'])
def index_page():
    if request.method == 'POST':
        # Get the user's query from the form
        query = request.form['query']

        # Create a query engine and get the response
        query_engine = index.as_query_engine(llm=llm)  # Explicitly pass HuggingFace LLM
        response = query_engine.query(query)
        
        # Render the result back to the frontend
        return render_template('index.html', query=query, response=str(response))

    # If GET request, show the form
    return render_template('index.html', query='', response='')

if __name__ == '__main__':
    app.run(debug=True)

