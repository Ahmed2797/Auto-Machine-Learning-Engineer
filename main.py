import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from rich.console import Console
from rich.syntax import Syntax
from rich.panel import Panel

from dotenv import load_dotenv
load_dotenv()

openai_api_key = os.getenv('OPENAI_API_KEY')
embeddings = OpenAIEmbeddings(model="text-embedding-3-small",api_key=openai_api_key)

def display_repo_context(query, context):
    console = Console()

    console.print(Panel(f"[bold green]Searching context for:[/bold green] {query}", expand=False))

    syntax = Syntax(
        context, 
        "python", 
        theme="monokai", 
        line_numbers=True,
        word_wrap=True
    )
    
    console.print(syntax)


def ask_agent_to_write_code(query):
    db = FAISS.load_local("faiss_index_code", embeddings, allow_dangerous_deserialization=True)
    
    relevant_docs = db.similarity_search(query, k=1)
    context = "\n".join([doc.page_content for doc in relevant_docs])
    
    print("Relevant context found from your repo!")
    return display_repo_context(query,context)

ask_agent_to_write_code('model_trainer',)