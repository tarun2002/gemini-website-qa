import os
import asyncio
import nest_asyncio
import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from my_api_key import key

nest_asyncio.apply()
if not asyncio.get_event_loop().is_running():
    asyncio.set_event_loop(asyncio.new_event_loop())


os.environ["GOOGLE_API_KEY"] = key 
os.environ["USER_AGENT"] = "my-streamlit-app/0.1"


st.title("🔎 Website Q&A with Gemini + LangChain")
st.write("Enter a website URL and ask a question. The app will fetch content, build embeddings, and answer.")

# User inputs
url = st.text_input("Enter Website URL", )
query = st.text_input("Enter your Question", )

if st.button("Get Answer"):
    with st.spinner("Fetching and processing..."):
        try:
            # 1. Load documents
            loader = WebBaseLoader(url)
            documents = loader.load()

            # 2. Split into chunks
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_documents(documents)

            # 3. Create vector DB
            embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
            vectorstore = FAISS.from_documents(chunks, embeddings)

            # 4. Create QA chain
            qa = RetrievalQA.from_chain_type(
                llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash"),
                retriever=vectorstore.as_retriever()
            )

            # 5. Run query
            answer = qa.invoke({"query": query})

            st.subheader("Answer:")
            st.write(answer["result"])

        except Exception as e:
            st.error(f"Error: {str(e)}")
