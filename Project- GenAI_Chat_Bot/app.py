from SearchService import chat_bot_response
import random
from warnings import filterwarnings
filterwarnings('ignore')

ENV = 'azure' ## azure or local
assistant_names = ['Chandler Bing', 'Joey Tribbiani', 'Ross Geller', 'Rachel Green', 'Monica Geller', 'Phoebe Buffay']
name = random.choice(assistant_names)

print(chat_bot_response(name= name, user_query="Introduce yourself please.", env = ENV))
while True:
    query = input("What do you want to know?\n")
    ai_response = chat_bot_response(name= name, user_query= query, env = ENV)
    print(ai_response)

