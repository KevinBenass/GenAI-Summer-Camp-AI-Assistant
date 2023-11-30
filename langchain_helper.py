import openai
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env (especially openai api key)

openai.api_key= os.environ['OPENAI_API_KEY']

# Create LLM model
llm =OpenAI(openai_api_key=os.environ['OPENAI_API_KEY'],temperature=0)
vectordb_file_path = "faiss_index"


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
