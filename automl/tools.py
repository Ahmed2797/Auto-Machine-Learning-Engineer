import os
import subprocess
import ast
from crewai.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_classic.schema import HumanMessage, SystemMessage
from dotenv import load_dotenv
load_dotenv()

openai_api_key = os.getenv('OPENAI_API_KEY')

@tool("list_empty_files")
def list_empty_files(directory: str):
    """Finds all code files that are empty or contain only the placeholder."""
    empty_files = []
    # Only look at these extensions to avoid binary errors
    valid_extensions = ('.py', '.js', '.txt', '.md', '.json', '.yaml', '.yml', '.html', '.css')
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if not file.endswith(valid_extensions):
                continue # Skip images, compiled code, etc.
                
            path = os.path.join(root, file)
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Check if it's the placeholder we created
                    if "# Protected AI Generation" in content or len(content.strip()) < 50:
                        # Return path relative to the SAFE_PROJECT_DIR for the agent
                        rel_path = os.path.relpath(path, SAFE_PROJECT_DIR)
                        empty_files.append(rel_path)
            except (UnicodeDecodeError, PermissionError):
                continue # Skip files that can't be read as text
                
    return empty_files


# Create the specific tool for the agent
@tool("generate_and_write_logic")
def generate_and_write_logic(file_path: str):
    """
    Search the original repo via FAISS, generates the code for a specific file path,
    and physically saves it to the replicated_project_output folder.
    Use this for empty files like 'data_ingestion.py' or 'model_trainer.py'.
    """
    # Use the logic you already wrote!
    # We use the filename as the query to find relevant snippets in FAISS
    query = f"Code logic and imports for {os.path.basename(file_path)}"
    
    # 1. FAISS Search (Using your existing embeddings/index)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small",api_key=openai_api_key)

    db = FAISS.load_local("faiss_index_code", embeddings, allow_dangerous_deserialization=True)
    relevant_docs = db.similarity_search(query, k=1)
    context = "\n".join([doc.page_content for doc in relevant_docs])
    
    # 2. LLM Generation
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0) # Using 4o for better logic
    system_prompt = f"Expert Developer. Write '{file_path}' matching this style: {context}. ONLY code."
    
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Write the complete code for {file_path}")
    ])
    
    # 3. Clean and Save to the SAFE directory
    clean_code = response.content.replace("```python", "").replace("```", "").strip()
    
    # Force the write into your replicated output folder
    full_path = os.path.join("./replicated_project_output", file_path.lstrip("./"))
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    
    with open(full_path, "w") as f:
        f.write(clean_code)
        
    return f"Successfully wrote code to {full_path}"


class RepoTool:
    @tool("get_full_file_list")
    def get_full_file_list(dummy_query: str):
        """Returns a complete list of all file paths stored in the FAISS index."""
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small",api_key=os.getenv("OPENAI_API_KEY") )
        db = FAISS.load_local("faiss_index_code", embeddings, allow_dangerous_deserialization=True)
        
        # Access the underlying docstore to get all metadata (file paths)
        all_paths = set()
        for doc_id in db.index_to_docstore_id.values():
            doc = db.docstore.search(doc_id)
            if 'source' in doc.metadata:
                # Clean the path to be relative to the new project
                path = doc.metadata['source'].replace("./my_downloaded_repo/", "")
                all_paths.add(path)
        
        return "\n".join(list(all_paths))

# Define your SAFE base directory
SAFE_PROJECT_DIR = "./replicated_project_output"

class FileSystemTool:
    @tool("create_directory_structure")
    def create_directory_structure(structure_input: str):
        """Creates folders and files ONLY inside the safe output directory."""
        # Convert input string to list
        try:
            import ast
            structure_list = ast.literal_eval(structure_input) if isinstance(structure_input, str) else structure_input
        except:
            return "Error: Input must be a list of paths."

        created = []
        for path in structure_list:
            # SAFETY: Force the path to be inside our SAFE_PROJECT_DIR
            # This prevents the agent from using "../../" to escape
            clean_path = os.path.join(SAFE_PROJECT_DIR, path.lstrip("/"))
            
            directory = os.path.dirname(clean_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
            
            if "." in os.path.basename(clean_path): # If it's a file
                with open(clean_path, "w") as f:
                    f.write("# Protected AI Generation\n")
                created.append(f"Safe File: {clean_path}")
            else:
                os.makedirs(clean_path, exist_ok=True)
                created.append(f"Safe Folder: {clean_path}")
                
        return "\n".join(created)

class DevelopmentTools:
    @tool("file_manager")
    def file_manager(path: str, content: str = None):
        """Writes code to files. Path should be relative (e.g., 'src/main.py')."""
        # Ensure we are always inside the safe directory
        full_path = os.path.join(SAFE_PROJECT_DIR, path.lstrip("/"))
        
        directory = os.path.dirname(full_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        
        if content:
            with open(full_path, "w") as f:
                f.write(content)
            return f"Successfully wrote code to {full_path}"
        return f"Created directory {directory}"

    @tool("execute_and_debug")
    def execute_and_debug(path: str):
        """Runs the python file inside the safe directory and returns errors."""
        full_path = os.path.join(SAFE_PROJECT_DIR, path.lstrip("/"))
        try:
            result = subprocess.run(['python3', full_path], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                return f"Success! Output: {result.stdout}"
            else:
                return f"Error Found:\n{result.stderr}"
        except Exception as e:
            return f"Execution failed: {str(e)}"