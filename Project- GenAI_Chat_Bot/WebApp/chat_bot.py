import streamlit as st
import requests
import json

FAST_APT_URL = "http://0.0.0.0:8504"
# Welcome Page Config
st.set_page_config(page_title= "AI-ASSISTANT", page_icon="🤖", initial_sidebar_state= "expanded")
st.subheader(body = f'{requests.post(url= f"{FAST_APT_URL}/get_response", data = json.dumps({"query":"Who are you?","messages":[]})).text}')


# initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# populate all session data
for message in st.session_state.messages:
    with st.chat_message(name = message["role"], avatar=message["avatar"]):
        st.markdown(message["content"])
    

# new query an response
if query := st.chat_input(placeholder="What do you want to know?"):
    with st.chat_message(name = "Human", avatar="🤔"):
        st.session_state["query"] = query
        # st.write(st.session_state.messages)
        # print(f"chat session : {query}")
        api = "get_response"
        
        user_input = dict(st.session_state)

        res = requests.post(url= f"{FAST_APT_URL}/{api}", data = json.dumps(user_input))

        st.session_state.messages.append({"role": "Human", "avatar":"🤔", "content":query})
        st.markdown(query)
    
    with st.chat_message(name="AI-Assistant", avatar="🤖"):
        st.session_state.messages.append({"role": "AI-Assistant", "avatar":"🤖", "content":res.text})
        st.markdown(res.text)
        # st.write(f"{res.text}")
    
    
        
    