from dotenv import load_dotenv

load_dotenv()
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers.web_research import WebResearchRetriever
from langchain_google_community import GoogleSearchAPIWrapper

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="chromaDB", embedding_function=embeddings)
search = GoogleSearchAPIWrapper()
web_research_retriever = WebResearchRetriever.from_llm(
    search=search,
    llm=llm,
    vectorstore=vectorstore,
)
