from dotenv import load_dotenv
import os
from langchain_pinecone import PineconeVectorStore
from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings

load_dotenv()

def main():
    print("Hello from langchain-course!")
    loader = UnstructuredLoader(
        file_path = "C:/Users/achus/Documents/langchain-course/mediumblog1.txt",
        chunking_strategy = "basic",
        max_characters = 1000000,
    )
    document = loader.load()

    text_splitter = CharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 0,
    )

    texts = text_splitter.split_documents(document)
    print(f"Number of chunks: {len(texts)}")

    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    print("Ingesting...")
    PineconeVectorStore.from_documents(texts, embeddings, index_name=os.environ.get("INDEX_NAME"))
    print("Ingestion complete!")


if __name__ == "__main__":
    main()