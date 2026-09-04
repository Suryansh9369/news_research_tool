import streamlit as st
import pickle
import time
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from dotenv import load_dotenv
load_dotenv()

st.title("News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls=[]

for i in range(3):
    url=st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked=st.sidebar.button("Process URLs")
# file_path="../notebooks/vector_index/index.pkl"
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", max_tokens=300)

main_placeholder=st.empty() # insert a single element container that allows you to replace or remove elements dynamically

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    output_dimensionality=768
)

if process_url_clicked:
    #load data
    loader=UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data=loader.load()
    
    # split data
    text_splitter=RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs=text_splitter.split_documents(data)
    # create embeddings and save it to FAISS index
    # embeddings=GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", output_dimensionality=768)
    
    vectorstore_geminiai=FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    # time.sleep(2)
    
    # Save the FAISS index to a pickle file
    vectorstore_geminiai.save_local("vector_index")
    
query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists("vector_index"):
        vectorstore=FAISS.load_local("vector_index", embeddings, allow_dangerous_deserialization=True)
        prompt = ChatPromptTemplate.from_template("""                                                  
            Answer the question using the provided context.

            Context:
            {context}

            Question:
            {question}
            """)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        chain = (
            {
                "context": retriever,
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
        )
        result = chain.invoke(query)

        st.header("Answer")
        st.write(result.content[0]["text"])
        