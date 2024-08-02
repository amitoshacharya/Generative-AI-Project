"""This module contains common functions used all over the projects."""

from langchain_core.prompts.prompt import PromptTemplate
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain, ConversationalRetrievalChain
from langchain.tools.retriever import create_retriever_tool
from genAI_models import llm

def generate_prompt_template(template:str) -> str:
    """Generates a prompt template for the system.

    Args:
        template (str): Raw Template for the prompt.

    Returns:
        str: Prompt Template
    """ 
    SYS_PROMPT = PromptTemplate(template= template, input_variables= ["context", "input_question"])
    return SYS_PROMPT


def invoke_LLMChain(model, prompt, user_question:str, chat_history:list[tuple]= [],
                  llm_chain:str = "default", verbose:bool = False, **chain_args):
    """Invokes or call the LLM chain to generate the answer for the input question.

    Args:
        model (LLM): LLM Model
        prompt (str): Raw Template for the prompt
        chat_history (list[tuple]): Chat History, list of tuples containing (question, answer) from the previous conversation.
        user_question (str): User Query
        llm_chain (str, optional): Chain Type. Defaults to "default".
                                    If "default", load_qa_with_sources_chain: QA over `documents directly` rather depending on retriever and `Answer Response from llm` include sources.
                                    If "RetrievalQA": QA over `retrieved documents`, for this retriever is required.
                                    If "RetrievalQAWithSourcesChain": QA over `retrieved documents`, for this retriever is required and `Answer Response from llm` include sources.
                                    If "ConversationalRetrievalChain": QA over `retrieved documents` and ongoing chat conversation (chat_history), for this retriever is required.
                                    If "load_qa_chain": If huge documents are there, then use this chain to load the documents and answer the question.
        verbose (bool, optional): Verbose. Defaults to False.

    Returns:
        str: Answer
    """
    chain_args = chain_args['chain_args'] if 'chain_args' in chain_args else {}
    prompt_template = generate_prompt_template(prompt)
    if chain_args:
        # try:
        if llm_chain == "default":
            chain_type = chain_args["chain_type"] if "chain_type" in chain_args else "stuff"
            chain = load_qa_with_sources_chain(llm = model,
                                               prompt = prompt_template,
                                               chain_type= chain_type,
                                               verbose = verbose
                                              )
            
            response = None
        
        elif llm_chain == "RetrievalQAWithSourcesChain":
            if "retriever" in chain_args:
                retriever = chain_args["retriever"]
            else:
                raise ValueError("Retriever is not provided for RetrievalQAWithSourcesChain")

            # chain_type = chain_args["chain_type"] if "chain_type" in chain_args else "stuff"
            chain = RetrievalQAWithSourcesChain.from_chain_type(llm = model,
                                                                retriever= retriever,
                                                                # chain_type= chain_type,
                                                                chain_type_kwargs= {'prompt': prompt_template},
                                                                return_source_documents= True,
                                                                verbose = verbose
                                                                )
            
            response = None
        
        elif llm_chain == "RetrievalQA":
            if "retriever" in chain_args:
                retriever = chain_args["retriever"]
            else:
                raise ValueError("Retriever is not provided for RetrievalQA")

            chain_type = chain_args["chain_type"] if "chain_type" in chain_args else "stuff"
            chain = RetrievalQA.from_chain_type(llm = model,
                                                retriever = retriever,
                                                chain_type= chain_type,
                                                chain_type_kwargs= {'prompt': prompt_template},
                                                return_source_documents= True,
                                                verbose = verbose
                                                )
            
            response = chain.invoke({"query":user_question})
            
        elif llm_chain == "ConversationalRetrievalChain":            
            if "retriever" in chain_args:
                retriever = chain_args["retriever"]
            else:
                raise ValueError("Retriever is not provided for ConversationalRetrievalChain")
            
            if chat_history:
                memory = ConversationBufferWindowMemory(memory_key="chat_history", output_key="answer", return_messages=True)
                memory.chat_memory.add_user_message(chat_history[0])
                memory.chat_memory.add_ai_message(chat_history[1])
            else:
                memory = ConversationBufferWindowMemory(memory_key="chat_history", output_key="answer", return_messages=True)

            chain = ConversationalRetrievalChain.from_llm(llm=model,
                                                            retriever=retriever,
                                                            memory= memory,
                                                            get_chat_history=lambda h : h,
                                                            return_source_documents = True,
                                                            combine_docs_chain_kwargs = {'prompt': prompt_template},
                                                            verbose= verbose
                                                            )
            response = chain.invoke({"question":user_question, "chat_history": chat_history})

        elif llm_chain == "load_qa_chain":
            chain = load_qa_chain(llm = model,
                                  chain_type= "map_reduce",
                                  verbose = verbose
                                  )
            response = chain({"input_documents": chain_args["input_documents"], "question": user_question},
                                    return_only_outputs=True)
            
          
        return {"llm_chain": llm_chain, "response": response}
        # except Exception as e:
        #     message = f"Error: {e}"
        #     print(message)
        #     return None
               
def get_retriever_agent(as_retriever, tool_name, tool_description,):
    """Get the retriever agent for the system.

    Args:
        model (LLM): LLM Model
        retriever (Retriever): Vector Store as a Retriever

    Returns:
        Retriever: Retriever Agent
    """

    ## Initialize Tool
    retriever_tool = create_retriever_tool(retriever = as_retriever,
                                           name = tool_name,
                                           description = tool_description)
    
    return retriever_tool
    
    ## Creating Agent Component: Agent
    # agent = 
    


    # return agent       