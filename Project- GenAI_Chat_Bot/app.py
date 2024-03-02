from SearchService import chat_bot_response
import WebApp as wp
import random
from warnings import filterwarnings
filterwarnings('ignore')
ENV = 'azure' ## azure or local
assistant_names = ['Chandler Bing', 'Joey Tribbiani', 'Ross Geller', 'Rachel Green', 'Monica Geller', 'Phoebe Buffay']
name = random.choice(assistant_names)

# Chat BOT Page Config
wp.set_page_config(title= "AI-Assistant", icon="🤖", initial_sidebar_state= "expanded")

first_response = chat_bot_response(name= name, user_query="Introduce yourself please.", env = ENV)
print(first_response)

wp.add_title(first_response)

# Chat BOT Page
bot_icon = "🤖"
while True:
    wp.write_text(text= "What do you want to know?")
    query = input("What do you want to know?\n")
    ai_response = chat_bot_response(name= name, user_query= query, env = ENV)

    wp.write_text(text= ai_response)
    print(ai_response)

