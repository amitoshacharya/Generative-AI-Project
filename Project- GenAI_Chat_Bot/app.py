from SearchService import chat_bot_response
from warnings import filterwarnings


filterwarnings('ignore')
print(chat_bot_response("Introduce yourself please."))
while True:
    query = input("What do you want to know? ")
    print(chat_bot_response(query))