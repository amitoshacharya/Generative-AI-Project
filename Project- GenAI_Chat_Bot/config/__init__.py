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
config ["OPEN_API_KEY"] = os.environ.get("OPEN_API_KEY")

if __name__ == "__main__":
    print(config)