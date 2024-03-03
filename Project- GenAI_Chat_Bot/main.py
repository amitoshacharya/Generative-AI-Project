import constants
from fastapi import FastAPI
import random
import schemas
from SearchService import chat_bot_response
import uvicorn
from warnings import filterwarnings
import json

filterwarnings('ignore')

env = constants.ENV
name = random.choice(constants.ASSISTANT_NAMES)

## Fast API
app = FastAPI()
PORT = constants.PORT


@app.post(path="/get_response")
def get_response(_input: schemas.UserInput):
    # response = chat_bot_response(chat_history=_input.messages, name = name, user_query= _input.query, env= env)
    response = chat_bot_response(name = name, user_query= _input.query, env= env)
    return response
    

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=True)
