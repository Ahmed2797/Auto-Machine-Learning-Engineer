import os
from git import Repo
import os
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import Language
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from rich.console import Console
from rich.syntax import Syntax
from rich.panel import Panel

from dotenv import load_dotenv
load_dotenv()

openai_api_key = os.getenv('OPENAI_API_KEY')

def clone_repository(repo_url, local_path):
    try:
        if not os.path.exists(local_path):
            print(f"Cloning {repo_url} into {local_path}...")
            Repo.clone_from(repo_url, local_path)
            print("Clone successful!")
        else:
            print("Folder already exists. Pulling latest changes...")
            repo = Repo(local_path)
            origin = repo.remotes.origin
            origin.pull()
            print("Update successful!")
            
    except Exception as e:
        print(f"An error occurred: {e}")


# উদাহরণ:
URL = "https://github.com/Ahmed2797/End-To-End-ML-Project"
DESTINATION = "./my_downloaded_repo"

clone_repository(URL, DESTINATION)



repo_path = "./my_downloaded_repo"



loader = GenericLoader.from_filesystem(path=repo_path,
                                       glob="**/*",
                                       suffixes=['.py'],
                                       parser=LanguageParser(language=Language.PYTHON,
                                                             parser_threshold=100))
documents = loader.load()

from langchain_text_splitters import RecursiveCharacterTextSplitter
python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size=6000, chunk_overlap=100
)
texts = python_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small",api_key=openai_api_key)
vector_db = FAISS.from_documents(texts, embeddings)
vector_db.save_local("faiss_index_code")

print("Repo success")



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