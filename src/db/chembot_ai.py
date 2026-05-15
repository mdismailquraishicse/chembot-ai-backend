import os
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS




class ChemDB:


    def __init__(self):

        self.embeddings = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")


    def handle_document(self, path_pdf:str, index:str = "faiss_index"):

        # Load pdf
        loader = PyMuPDFLoader(path_pdf)
        document = loader.load()

        # Split text into chunks
        text_splitter = CharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 100
        )
        chunks = text_splitter.split_documents(document)

        # Convert chunks into embeddings and save locally
        db = FAISS.from_documents(chunks, self.embeddings)
        db.save_local(index)
        print(f"pdf converted into embeddings and saved to: {index}")
        return True

    
    def get_faiss(self, path_pdf, index_path="faiss_index"):

        print(f"loading faiss db...")
        if not os.path.exists(f"{index_path}/index.faiss"):
            print(f"Embeddings not exists creating...")
            self.handle_document(path_pdf= path_pdf, index= index_path)

        db = FAISS.load_local(index_path,
                                self.embeddings,
                                allow_dangerous_deserialization = True)

        return db