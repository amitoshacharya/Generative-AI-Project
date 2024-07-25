"""This file is used to test the LangChain Agents for the user query."""

from constants import vectorDB_path
from langchain.agents import AgentExecutor, create_react_agent
from helper import get_retriever_agent
from vectorStore_embeddings import get_retriever


if __name__ == "__main__":
    session_start = True
    
    while True:
        ## user input
        # query = input("Enter your query: ")
        query = "I have a mild fever and also headache. Please suggest me some test that I should undergo."

        ## Retriever
        search_type = "similarity" ## "similarity" or "mmr"
        search_kwargs = {"k":2, "score_threshold": 0.95}
        retriever = get_retriever(vectorDB_path = vectorDB_path,
                                  search_type = search_type,
                                  search_kwargs = search_kwargs)
        
        # Tool
        retriever_tool = get_retriever_agent(as_retriever = retriever,
                                             tool_name = "medical_instruction_retriever",
                                             tool_description = "Search and retrieve medical instructions for the user query.")
        
        # print(retriever_tool.func)
        print(retriever_tool.invoke({"query": query }))












        break