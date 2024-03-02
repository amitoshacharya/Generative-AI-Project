import streamlit as st

def set_page_config(title:str=None, icon:str=None, layout:str="centered", initial_sidebar_state:str="auto", menu_items:str=None):
    """
    This function add configuration to the app.
    """
    return st.set_page_config(
        page_title=title, 
        page_icon=icon, 
        layout= layout,
        initial_sidebar_state = initial_sidebar_state,
        menu_items = menu_items
        )

def add_title(title:str):
    """
    This function add title to the page content.
    """
    return st.title(title)

def sub_header(text:str=None):
    return st.subheader(body= text)

def write_text(text:str=None):
    return st.write(text)

def input_text(label:str= None, value=""):
    pass

