# **Project Scenario**
- ## **OBJECTIVE** *(Version Categorize)*
    - V1
        - Create a streamlit app which is a chatbot.
        - This chatbot takes user's querie and process to generate responses under 50 words.
        - Chatbot response should be breaked into bullet points for better understanding.

- ## Project Stage Description
    - ### **Streamlit APIs**
        - *app.py*
            - *Streamlit multipages WebAPI.*
            - *This WebAPI is an interactive user interface (UI) which connect frontend users to backened processing Large Language Model (LLM).*

    - ### **Chat Bot**
        - *SearchService*
            - *This is a backened functions that search for the response to user's asked queries.*

    - **utils**
        - *prompt_template.py*
            - *This is a backened functions that sets-up prompts for LLM model.*

    

- ## **Execution Steps**
    - ### **Open GitHub Codespace**
        - *Click on "<> code"*
        - *Click on codespace.*
        - *Click on created codespace to open and proceed.*
    - ### **Creating Virtual Environment .venv**
        > python -m venv /designated_path
    - ### **Activating Environment**
        > source ./venv/bin/activate
    - ### **De-Activating Environment**
        > deactivate
    - ### **Entering To Project- GenAI_Chat_Bot**
        > cd "./Project- GenAI_Chat_Bot"
    - ### **Install Requirements**
        > pip install -r requirements.txt
    - ### **Creating Modules**
        - Add a folder and name it. For example: xyz
        - open terminal and type command to generate *\__init__.py*
            > *touch xyz/\__init__.py*
        - Now we can import xyz module in *app.py*
            > *import xyx*
        
    

