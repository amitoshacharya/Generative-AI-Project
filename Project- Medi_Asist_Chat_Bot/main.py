"""This file is used to test the LangChain Chains for the user query."""

from datetime import datetime
from constants import vectorDB_path, bot_instruction_template, chat_history_bot_instruction_template
from genAI_models import llm
from helper import invoke_LLMChain
from vectorStore_embeddings import get_retriever
from langchain_community.callbacks import get_openai_callback


if __name__ == "__main__":
    session_start = True
    
    while True:
        ## user input
        query = input("Enter your query: ")
        # query = "I have a mild fever and also headache. Please suggest me some test that I should undergo."
        # Retriever
        search_type = "similarity" ## "similarity" or "mmr"
        search_kwargs = {"k":2, "score_threshold": 0.95}
        retriever = get_retriever(vectorDB_path = vectorDB_path,
                                  search_type = search_type,
                                  search_kwargs = search_kwargs)
        
        # relevant_docs = retriever.invoke(query)
        # print(f"Relevant Documents for the user query: '{query}' retrieved are {len(relevant_docs)} ", end="\n"*2)
        # for doc in relevant_docs:
        #     print(doc.page_content)
        #     print("\n\n")

        # Invoking LLM Chain
        chain_args = {
            "chain_type": "stuff",
            "retriever": retriever
        }

        # Here we can RetrievalQA and ConversationalRetrievalChain
        if session_start:
            llm_chain = "RetrievalQA"
            history = []
            session_start = False
        else:
            llm_chain = "ConversationalRetrievalChain"

        if llm_chain == "RetrievalQA":
            LLM_PROMPT = bot_instruction_template
        elif llm_chain == "ConversationalRetrievalChain":
            LLM_PROMPT = chat_history_bot_instruction_template
        else:
            raise ValueError(f"Invalid LLM Chain: {llm_chain}. Please provide a valid LLM Chain.")

        verbose = False # to view the inputs to the LLM
        with get_openai_callback() as cb:
            print(f"LLM Cost Chart for --> {datetime.now().strftime('%D  %H:%M:%S %p')}\n{'-'*20}")
            chain_response = invoke_LLMChain(model = llm,
                                    prompt = LLM_PROMPT,
                                    llm_chain = llm_chain,
                                    chain_args = chain_args,
                                    user_question = query,
                                    chat_history = history,
                                    verbose = verbose
                                    )
            print(f"Used Tokens in LLM call: {cb.total_tokens}")
            print(f"Spent Cost (USD) in LLM call: ${cb.total_cost}")


        if chain_response['llm_chain'] == "RetrievalQA":
            user_question = chain_response['response']['query']
            chat_history = "No Chat History due to session start."
            answer = chain_response['response']['result']

        elif chain_response['llm_chain'] == "ConversationalRetrievalChain":
            user_question = chain_response['response']['question']
            chat_history = chain_response['response']['chat_history']
            answer = chain_response['response']['answer']

        print(f"User Question: {user_question}", end="\n"*2)
        print(f"Chat History: {chat_history}", end="\n"*2)
        print(f"Answer: {answer}", end="\n"*2)


        ## Source Documents
        return_source = False
        if return_source:
            print(f"Source Documents:", end="\n"*2)
            source_docs = chain_response['response']['source_documents']
            for doc in source_docs:
                metadata = doc.metadata
                source = metadata['source'].split('\\')[-1]
                page = metadata['page']
                print(f"Document Name: {source} and Page No.: {page}")

                print("Source page content: \n")
                print(doc.page_content)
                
                print("\n"*3)
        
        history = [user_question, answer]        