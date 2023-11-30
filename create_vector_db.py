from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema.document import Document

# # Initialize instructor embeddings using the Hugging Face model
instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
vectordb_file_path = "faiss_index"
context_file_path = "context.txt"


def get_context(context_file_path):
    with open(context_file_path, 'r') as file:
        # Read the content of the file into a string
        context = file.read()
    return context


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = [Document(page_content=x) for x in text_splitter.split_text(text)]
    return docs


def create_vector_db():
    str_context = get_context(context_file_path)
    docs = get_text_chunks(str_context)

    # Create a FAISS instance for vector database from docs
    vectordb = FAISS.from_documents(documents=docs,
                                    embedding=instructor_embeddings)

    # Save vector database locally
    vectordb.save_local(vectordb_file_path)

if __name__ == "__main__":
    create_vector_db()
