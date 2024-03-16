"""
This function helps to set prompts.
"""
from langchain.prompts import PromptTemplate

def chat_response_template():
    template = """
    You are an AI assistance named {name}, who responds to user's queries under 50 words.
    ###
    User: {query}
    AI Assistant:
    """

    # add Information :{context} and input_variables "context"
    prompt_template = PromptTemplate(template= template, input_variables=['name','query']) 
    return prompt_template