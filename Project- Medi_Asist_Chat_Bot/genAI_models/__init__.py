"""This module initialize the genAI models for chat and embeddings."""

from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()
########################## Azure OpenAI Config #############################
CREDS_OF = os.getenv("CREDS_OF")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION")
OPENAI_ENDPOINT = os.getenv("OPENAI_ENDPOINT")
OPENAI_DEPLOYMENT = os.getenv("OPENAI_DEPLOYMENT")
OPENAI_MODEL = os.getenv("OPENAI_MODEL")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL")
OPENAI_TYPE = os.getenv("OPENAI_TYPE")


########################## Azure Chat OpenAI LLM Model #############################
llm = AzureChatOpenAI(
        temperature=0,
        api_key=OPENAI_API_KEY,
        api_version=OPENAI_API_VERSION,
        azure_endpoint=OPENAI_ENDPOINT,
        azure_deployment=OPENAI_DEPLOYMENT,
        model=OPENAI_MODEL,          
        # model_kwargs= { "top_p": 1}
    )

########################## Azure OpenAI Embedding Model #############################
embeddings = AzureOpenAIEmbeddings(
    model=OPENAI_EMBEDDING_MODEL, 
    api_key=OPENAI_API_KEY, 
    azure_endpoint=OPENAI_ENDPOINT, 
    disallowed_special=(),
    )


if __name__ == "__main__":
    # os.environ["HTTPS_PROXY"]="blrproxy.ad.infosys.com:443"
    # os.environ["HTTP_PROXY"]="blrproxy.ad.infosys.com:443"
    # try:
    print(f"Working with Credentials: {CREDS_OF}")
    print(llm)
    print(llm.invoke("Hi"))
    input_text = input("Question: ")
    print(llm.invoke(input_text))
    # except Exception as e:
    #     print(f"Error: {e}")
    #     print(f"Error: {e.__traceback__}")