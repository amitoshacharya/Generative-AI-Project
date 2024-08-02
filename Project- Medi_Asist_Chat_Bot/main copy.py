"""This file is used to test the LangChain Chains for the user query with chain type map_reduce."""

from datetime import datetime
from constants import vectorDB_path, bot_instruction_template, chat_history_bot_instruction_template, local_dir_path
from genAI_models import llm
from helper import invoke_LLMChain
from vectorStore_embeddings import get_retriever
from langchain_community.callbacks import get_openai_callback
from vectorStore_embeddings import load_documents


if __name__ == "__main__":
    session_start = True
    
    while True:
        ## user input
        # query = input("Enter your query: ")
        query = "I have a mild fever and also headache. Please suggest me some test that I should undergo."
        # Retriever
        search_type = "similarity" ## "similarity" or "mmr"
        search_kwargs = {"k":2, "score_threshold": 0.95}
        
        documents = load_documents(local_dir_path)
        
        res = invoke_LLMChain(llm_chain="load_qa_chain",
                              model=llm,
                              prompt= bot_instruction_template,
                              chain_args={"input_documents": documents[:5]},
                              user_question=query,
                              verbose=True
                              )
        print(res)