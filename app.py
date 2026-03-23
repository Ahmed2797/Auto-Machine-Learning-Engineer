from automl.tools import *

from crewai import Agent, Task, Crew, Process

# 1. The Architect: Mirrors the structure
architect = Agent(
    role='System Architect',
    goal='Replicate the folder structure of the source repository physically on disk',
    backstory="""You are an expert at mapping project layouts. You don't just talk; you 
    use your tools to build the actual folders and files on the user's computer.""",
    # Use the tool that handles directory lists!
    tools=[RepoTool.get_full_file_list, FileSystemTool.create_directory_structure], 
    verbose=True,
    allow_delegation=True
)

# 1.0. Defining Tasks
task_structure = Task(
    description=f"""
    1. Use 'get_full_file_list' to see the source structure.
    2. Use 'create_directory_structure' to build the SAME structure.
    3. IMPORTANT: All files must be created inside '{SAFE_PROJECT_DIR}'. 
    4. Do NOT modify or delete anything in the source folder.
    """,
    agent=architect,
    expected_output="A complete replica of the folder structure in a separate, safe directory."
)


# 2. Write code agent
developer = Agent(
    role='Software Implementation Engineer',
    goal='Fill every empty file in ./replicated_project_output with logic from the FAISS index.',
    backstory="""You are a precision engineer. You use 'list_empty_files' to find work.
    You then use 'generate_and_write_logic' to physically save the code.
    Finally, you use 'execute_and_debug' to verify the file works.""",
    tools=[
        list_empty_files,
        generate_and_write_logic,
        DevelopmentTools.execute_and_debug
    ],
    verbose=True
)

# write code task
implementation_task = Task(
    description="""
    1. **Audit:** Call 'list_empty_files' for './replicated_project_output'.
    2. **Implement:** For every file in that list:
       a. Call 'generate_and_write_logic' with the file path.
       b. Call 'execute_and_debug' on that file.
       c. If it fails, call 'generate_and_write_logic' again with the error to fix it.
    3. **Finish:** Stop only when 'list_empty_files' returns [].
    """,
    agent=developer,
    expected_output="All files physically saved and verified on disk."
)

## crew
project_crew = Crew(
    agents=[architect,developer],
    tasks=[task_structure,implementation_task],
    process=Process.sequential,
    verbose=True
)

project_crew.kickoff()