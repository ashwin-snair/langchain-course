import os
from operator import itemgetter
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_pinecone import PineconeVectorStore
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

print("Initializing  components...")

embeddings = OllamaEmbeddings(model="nomic-embed-text")
llm = ChatOllama(model="qwen3:1.7b", temperature=0.2, max_tokens=512)

vectorstore = PineconeVectorStore(
    index_name=os.environ.get("INDEX_NAME"),
    embedding=embeddings,
)

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

prompt_template = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context

    {context}

    Question: {question}

    Provide a detailed answer;
    """
)

def format_docs(docs):
    """Format retrieved documents into a single string"""
    return "\n\n".join(doc.page_content for doc in docs)

def retrieval_chain_without_lcel(query: str):
    docs = retriever.invoke(query)
    context = format_docs(docs)
    messages = prompt_template.format_messages(context = context, question = query)

    response = llm.invoke(messages)

    return response.content

def retrieval_chain_with_lcel():
    '''
    Create a retrieval chain using  LCEL (Langchain Expression Language).
    Returns a chain that can be invoked with {"question": "..."}
    '''
    retrieval_chain = (
        RunnablePassthrough.assign(
            context = itemgetter("question") | retriever | format_docs
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )

    return retrieval_chain

if __name__ == "__main__":
    print("Retrieving")

    query = "What is pinecone in machine learning?"

    # ========================================================
    # Option 1: Retrieval without LCEL
    # ========================================================
    print("\nRetrieval without LCEL")
    results_without_lcel = retrieval_chain_without_lcel(query)
    print("\nAnswer:")
    print(results_without_lcel)

    #  ========================================================
    # Otion 2: Retrieval with LCEL
    # ========================================================

    print("\nRetrieval with LCEL")
    chain_with_lcel =  retrieval_chain_with_lcel()
    results_with_lcel = chain_with_lcel.invoke({"question": query})
    print("\nAnswer:")
    print(results_with_lcel)



