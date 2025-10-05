import langchain
from langchain.chat_models import ChatOllama
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

def vector_DB():
    llm = OllamaEmbeddings(model = 'llama3')
    document = TextLoader("ml.txt").load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100,chunk_overlap=10)
    chunk = text_splitter.split_documents(document)
    db = Chroma.from_documents(chunk,llm)
    retriever = db.as_retriever()

    while True:
        text = input("ask the question")
        if text == "end":
            break
        ans = retriever.invoke(text)

        for a in ans:
            print(a.page_content)
