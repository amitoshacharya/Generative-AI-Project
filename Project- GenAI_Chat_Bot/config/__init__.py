"""
This file Contains configuration of secrets keys.
To use this module, import in other modules using: 
    from config import config
"""
import os
from dotenv import load_dotenv

## loading .env file for secrets
load_dotenv()

## public variable to import config to other modules
config = {}

## LOCAL
config["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY")

## AZURE
config["AZURE_OPENAI_API_KEY"] = os.environ.get("AZURE_OPENAI_API_KEY")
config["AZURE_OPENAI_API_VERSION"] = os.environ.get("AZURE_OPENAI_API_VERSION")
config["AZURE_OPENAI_API_DEPLOYMENT"] = os.environ.get("AZURE_OPENAI_API_DEPLOYMENT")
config["AZURE_OPENAI_API_KEY"] = os.environ.get("AZURE_OPENAI_API_KEY")
config["AZURE_OPENAI_API_ENDPOINT"] = os.environ.get("AZURE_OPENAI_API_ENDPOINT")

if __name__ == "__main__":
    print(config)