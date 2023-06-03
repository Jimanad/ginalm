import re
import openai
import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv
from streamlit_option_menu import option_menu
import streamlit.components.v1 as html
from  PIL import Image
import numpy as np
#from  PIL import ImageChops
import pandas as pd
import plotly.express as px
import io 
import streamlit.components.v1 as components
import plantuml

load_dotenv('.env')
openai.api_key = "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"
model_name = "GPT-3.5"

# Map model names to OpenAI model IDs
if model_name == "GPT-3.5":
    model = "gpt-3.5-turbo"
else:
    model = "gpt-4"

# Setting page title and header
st.set_page_config(page_title="ALM", page_icon=":robot_face:")
st.markdown("<h1 style='text-align: center;'>ALM - Application Lifecycle Management</h1>", unsafe_allow_html=True)


# Initialise session state variables
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []
if 'messages' not in st.session_state:
    st.session_state['messages'] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
if 'model_name' not in st.session_state:
    st.session_state['model_name'] = []
if 'cost' not in st.session_state:
    st.session_state['cost'] = []
if 'total_tokens' not in st.session_state:
    st.session_state['total_tokens'] = []
if 'total_cost' not in st.session_state:
    st.session_state['total_cost'] = 0.0



# generate a response
def generate_response(prompt):
    st.session_state['messages'].append({"role": "user", "content": prompt})

    completion = openai.ChatCompletion.create(
        model=model,
        messages=st.session_state['messages']
    )
    response = completion.choices[0].message.content
    st.session_state['messages'].append({"role": "assistant", "content": response})

    # print(st.session_state['messages'])
    total_tokens = completion.usage.total_tokens
    prompt_tokens = completion.usage.prompt_tokens
    completion_tokens = completion.usage.completion_tokens
    return response, total_tokens, prompt_tokens, completion_tokens

def generate_diagram(plantuml_code):
    diagram = plantuml.PlantUML(url='http://www.plantuml.com/plantuml/img/',
                          basic_auth={},
                          form_auth={}, http_opts={}, request_opts={})
    return diagram.processes(plantuml_code)

# Sidebar - let user choose model, show total cost of current conversation, and let user clear the current conversation
with st.sidebar:
    choose = option_menu("ALM", ["Requirement", "Usecase specification", "Usecase model", "ER diagram", "System context diagram", "Component model diagram", "Code","KB article","Runbook"],
                         icons=["Important", "Important","Important", "Important","Important", "Important","Important", "Important","Important"],
                         menu_icon="app-indicator", default_index=0,
                         styles={
        "container": {"padding": "5!important", "background-color": "#fafafa"},
        "icon": {"color": "orange", "font-size": "25px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#02ab21"},
    }
    )

if choose == "Requirement":

    # container for chat history
    response_container = st.container()
    # container for text box
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_area("You:", key='input', height=100)
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            output, total_tokens, prompt_tokens, completion_tokens = generate_response(user_input)
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)
            st.session_state['model_name'].append(model_name)
            st.session_state['total_tokens'].append(total_tokens)

            # from https://openai.com/pricing#language-models
            if model_name == "GPT-3.5":
                cost = total_tokens * 0.002 / 1000
            else:
                cost = (prompt_tokens * 0.03 + completion_tokens * 0.06) / 1000

            st.session_state['cost'].append(cost)
            st.session_state['total_cost'] += cost

    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
                message(st.session_state["generated"][i], key=str(i))
                st.write(
                    f"Model used: {st.session_state['model_name'][i]}; Number of tokens: {st.session_state['total_tokens'][i]}; Cost: ${st.session_state['cost'][i]:.5f}")
            
        
elif choose == "Usecase specification":
    st.markdown(""" <style> .font {
    font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
    </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">Use Case Specification</p>', unsafe_allow_html=True)
    prompt = "Generate use case specification for the above requirements";
    output, total_tokens, prompt_tokens, completion_tokens = generate_response(prompt)
    st.session_state['past'].append(prompt)
    st.session_state['generated'].append(output)
    st.session_state['model_name'].append(model_name)
    st.session_state['total_tokens'].append(total_tokens)

    # from https://openai.com/pricing#language-models
    if model_name == "GPT-3.5":
        cost = total_tokens * 0.002 / 1000
    else:
        cost = (prompt_tokens * 0.03 + completion_tokens * 0.06) / 1000

    st.session_state['cost'].append(cost)
    st.session_state['total_cost'] += cost
    response_container = st.container()
    if st.session_state['generated']:
        with response_container:
            st.write(f"{output}")
            

elif choose == "Usecase model":
    st.markdown(""" <style> .font {
    font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
    </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">Use Case Model</p>', unsafe_allow_html=True)

    prompt = "Generate plantUML script for usecase model diagram for the above usecase specification. Do not include any explanation other than the plantUML script in the response";
    umoutput, total_tokens, prompt_tokens, completion_tokens = generate_response(prompt)
    st.session_state['model_name'].append(model_name)
    st.session_state['total_tokens'].append(total_tokens)

    # from https://openai.com/pricing#language-models
    if model_name == "GPT-3.5":
        cost = total_tokens * 0.002 / 1000
    else:
        cost = (prompt_tokens * 0.03 + completion_tokens * 0.06) / 1000

    
    # initializing substrings
    sub1 = "@startuml"
    sub2 = "@enduml"
 
    # getting index of substrings
    idx1 = umoutput.index(sub1)
    idx2 = umoutput.index(sub2)
 
    # length of substring 1 is added to
    # get string from next character
    res = umoutput[idx1 : idx2]    
    
    st.session_state['cost'].append(cost)
    st.session_state['total_cost'] += cost
    response_container = st.container()
    if st.session_state['generated']:
        image = generate_diagram(res)
        st.image(image, use_column_width=True)
    

elif choose == "ER diagram":
     st.markdown(""" <style> .font {
    font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
    </style> """, unsafe_allow_html=True)
     st.markdown('<p class="font">ER Model diagram</p>', unsafe_allow_html=True)

     prompt = "Generate plantUML script for ER model diagram for the above usecase specification. Do not include any explanation other than the plantUML script in the response";
     eroutput, total_tokens, prompt_tokens, completion_tokens = generate_response(prompt)
     st.session_state['model_name'].append(model_name)
     st.session_state['total_tokens'].append(total_tokens)

    # from https://openai.com/pricing#language-models
     if model_name == "GPT-3.5":
        cost = total_tokens * 0.002 / 1000
     else:
        cost = (prompt_tokens * 0.03 + completion_tokens * 0.06) / 1000

     
     # initializing substrings
     sub1 = "@startuml"
     sub2 = "@enduml"
 
    
     st.session_state['cost'].append(cost)
     st.session_state['total_cost'] += cost
     response_container = st.container()
     if st.session_state['generated']:
         # getting index of substrings
         idx1 = eroutput.index(sub1)
         idx2 = eroutput.index(sub2)
    
        # length of substring 1 is added to
        # get string from next character
         res = eroutput[idx1 : idx2]    
         image = generate_diagram(res)
         st.image(image, use_column_width=True)
     
elif choose == "System context diagram":
    st.markdown(""" <style> .font {
    font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
    </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">System context diagram</p>', unsafe_allow_html=True)

    prompt = "Generate plantUML script for the system context diagram for the above usecase specification. Do not include any explanation other than the plantUML script in the response";
    scoutput, total_tokens, prompt_tokens, completion_tokens = generate_response(prompt)
    st.session_state['model_name'].append(model_name)
    st.session_state['total_tokens'].append(total_tokens)

    # from https://openai.com/pricing#language-models
    if model_name == "GPT-3.5":
        cost = total_tokens * 0.002 / 1000
    else:
        cost = (prompt_tokens * 0.03 + completion_tokens * 0.06) / 1000

    # initializing substrings
    sub1 = "@startuml"
    sub2 = "@enduml"
 
    # getting index of substrings
    idx1 = scoutput.index(sub1)
    idx2 = scoutput.index(sub2)
 
    # length of substring 1 is added to
    # get string from next character
    res = scoutput[idx1 : idx2]    
    
    st.session_state['cost'].append(cost)
    st.session_state['total_cost'] += cost
    response_container = st.container()
    if st.session_state['generated']:
      image = generate_diagram(res)
      st.image(image, use_column_width=True)
     

elif choose == "Component model diagram":
    st.markdown(""" <style> .font {
    font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
    </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">Component diagram</p>', unsafe_allow_html=True)

    prompt = "Generate plantUML script for the component model diagram for the above usecase specification. Do not include any explanation other than the plantUML script in the response";
    cdoutput, total_tokens, prompt_tokens, completion_tokens = generate_response(prompt)
    st.session_state['model_name'].append(model_name)
    st.session_state['total_tokens'].append(total_tokens)

    # from https://openai.com/pricing#language-models
    if model_name == "GPT-3.5":
        cost = total_tokens * 0.002 / 1000
    else:
        cost = (prompt_tokens * 0.03 + completion_tokens * 0.06) / 1000

    # initializing substrings
    sub1 = "@startuml"
    sub2 = "@enduml"
 
    # getting index of substrings
    idx1 = cdoutput.index(sub1)
    idx2 = cdoutput.index(sub2)
 
    # length of substring 1 is added to
    # get string from next character
    res = cdoutput[idx1 : idx2]    
    
    st.session_state['cost'].append(cost)
    st.session_state['total_cost'] += cost
    response_container = st.container()
    if st.session_state['generated']:
      image = generate_diagram(res)
      st.image(image, use_column_width=True)
    

elif choose == "KB article":
    st.markdown(""" <style> .font {
    font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
    </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">Knowledge article</p>', unsafe_allow_html=True)


elif choose == "Runbook":
    st.markdown(""" <style> .font {
    font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
    </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">Runbook</p>', unsafe_allow_html=True)

   

clear_button = st.sidebar.button("Clear Conversation", key="clear")

# Map model names to OpenAI model IDs
if model_name == "GPT-3.5":
    model = "gpt-3.5-turbo"
else:
    model = "gpt-4"

# reset everything
if clear_button:
    st.session_state['generated'] = []
    st.session_state['past'] = []
    st.session_state['messages'] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
    st.session_state['number_tokens'] = []
    st.session_state['model_name'] = []
    st.session_state['cost'] = []
    st.session_state['total_cost'] = 0.0
    st.session_state['total_tokens'] = []
    



