import openai
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env (especially openai api key)

vectordb_file_path = "faiss_index"
router_prompt_file_path = "ROUTER_PROMPT.txt"
application_prompt_file_path = "APPLICATION_PROMPT.txt"
question_prompt_file_path="QUESTION_PROMPT.txt"

openai.api_key= os.environ['OPENAI_API_KEY']

# Create LLM model
llm =OpenAI(openai_api_key=os.environ['OPENAI_API_KEY'],temperature=0)

def get_completion(prompt, model="gpt-3.5-turbo", temperature=0):
    response = openai.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": "You are a helpful assistant"},
                  {"role": "user", "content": prompt}],
        temperature=temperature,  # this is the degree of randomness of the model's output
        max_tokens=1024
    )
    return response.choices[0].message.content

def get_intent(query):
    router_prompt_template = get_text(router_prompt_file_path)
    prompt = router_prompt_template.format(query=query)
    response = get_completion(prompt)
    return response


def get_application_details(query, context,model="gpt-3.5-turbo", temperature=0):
    application_prompt_template = get_text(application_prompt_file_path)
    prompt = application_prompt_template.format(context_data=context, query=query)
    response = get_completion(prompt, model=model, temperature=temperature)
    return response
     
def get_retrieval_qa_chain():
     # Initialize instructor embeddings using the Hugging Face model
    instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
    # Load the vector database from the local folder
    vectordb = FAISS.load_local(vectordb_file_path, instructor_embeddings)

    # Create a retriever for querying the vector database
    retriever = vectordb.as_retriever(score_threshold=0.7)

    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type="stuff",
                                        retriever=retriever,
                                        input_key="query",
                                        return_source_documents=True,
                                        chain_type_kwargs={"prompt": PROMPT})

    return chain

if __name__ == "__main__":
    chain = get_retrieval_qa_chain()
    print(chain("What is the cost for the Summer Camp?")['result'])
