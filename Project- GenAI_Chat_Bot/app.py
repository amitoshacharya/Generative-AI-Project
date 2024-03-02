from SearchService import chat_bot_response
from warnings import filterwarnings

global ENV 

ENV = 'azure' ## azure or local
filterwarnings('ignore')
print(chat_bot_response("Introduce yourself please.", ENV))
while True:
    query = input("What do you want to know? ")
    ai_response = chat_bot_response(user_query=query, env = ENV)
    print(ai_response)

