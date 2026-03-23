import os
from git import Repo
import os
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import Language
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


from dotenv import load_dotenv
load_dotenv()

openai_api_key = os.getenv('OPENAI_API_KEY')
DESTINATION = "./my_downloaded_repo"

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

def load_repo():

    loader = GenericLoader.from_filesystem(path=DESTINATION,
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

    return print("Repo success")

if __name__ == '__main__':
    URL = "https://github.com/Ahmed2797/End-To-End-ML-Project"
    clone_repository(URL, DESTINATION)
    load_repo()