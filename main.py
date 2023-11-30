import streamlit as st
from langchain_helper import get_intent, get_answer, get_application_details

st.title("GenAI Conversational AI Assistant :computer:")

query = st.text_input(
    "Hi there! I'm Genny — GenAI Summer camp conversational AI assistant! I'd be happy to assist you. How can I help?")
if query:
    intent = get_intent(query)
    if intent == 'inquiry':
        response = get_retrieval_qa_chain()(query)
        st.header("Answer:")
        st.write(response)
    else:
        name=st.text_area("""Great! You are ready to enroll your child to the GenAI Summer Camp! You took the right decision!
                        To complete the registration, I would need you to provide me with your full name, your phone number, your email, and your child's age.
                        What is your full name?      Please type Command + Enter to confirm your input""")
        if name:
            phone = st.text_area("What is your phone number?")
            if phone:
                email = st.text_area("What is your email?")
                if email:
                    age = st.text_area("What is your child's age?")
                    if age:
                        response = get_application_details(age)
                        st.header("Answer:")
                        st.write(response)
