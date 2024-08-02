# Project Creation Flow

## Project Pre-Execution Steps
- **Virtual env.**
    ```
    python -m venv ./chatvenv
    ```
- **Activate Virtual env.**
    ```
    BASH COMMAND:
    source ./chatvenv/bin/activate
    ```
    ```
    CMD PROMPT:
    .\chatvenv\Scripts\activate 
    ```
- **To create __init__.py in folder to use it as package**
    ```
    BASH COMMAND:
    touch <folder_name>/__init__.py
    ```
- **To execute python file**
    ```
    python <file_name>.py
    ```
- **To install requirements.txt**
    ```
    pip install -r requirements.txt
    ```
## Project Description
- **Project 1** &rarr; Med-Assist
- **Objective** &rarr; A user interactive chat bot that helps the user to assist them with there medical emergencies.

## Project Content
### ***Readme***
- This file contains project files descriptions.

### ***Notes***
- This files contains required base conceptual knowledge.

### ***Save_text***
- This file contains the interesting use cases that are encountered during this project execution

### ***requirements***
- This is a text file, listing the pre-requisites for the project. 

### ***.env***
- This module contains the environment variables for the project.

### ***genAI_models***
- This module initialize the genAI models for chat and embeddings.

### ***vectorStore_embeddings***
- This module create the embeddings from documents using the genAI models.

### ***VectorDB-(FAISS)***
- FAISS Vector Database to store vector or embeddings of document for RAG based Application.

### ***constants***
- This module contains constants all over the project.

### ***helper***
- This module contains common functions used all over the projects

### ***documents***
- This folder contains the source document.

### ***main***
- This file is used to test the LangChain Chains for the user query.

### ***main_agents***
- This file is used to test the LangChain Agents for the user query.

### ***main copy***
- This file is used to test the LangChain Chains for the user query with chain type map_reduce.