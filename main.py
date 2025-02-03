from utils import vectorstore, web_research_retriever, llm
from langchain import hub
from langchain_core.runnables import chain


DEFAULT_THRESHOLD = 0.5

prompt = hub.pull("rlm/rag-prompt")


@chain
def retrieve_documents(question):
    retrieved_docs = vectorstore.similarity_search_with_relevance_scores(
        question, score_threshold=DEFAULT_THRESHOLD
    )

    if len(retrieved_docs) == 0:
        retrieved_docs = web_research_retriever.invoke(question)

    sources = []

    for item in retrieved_docs:
        if isinstance(item, tuple):
            sources.append(item[0].metadata["source"])
        else:
            sources.append(item.metadata["source"])

    print(sources)
    print(retrieved_docs)
    return {
        "question": question,
        "context": retrieved_docs,
        "sources": sources,
    }


@chain
def empty(sources):
    return {"sources": sources["sources"]}


chain = retrieve_documents | {"main": prompt | llm, "sources": empty}

# print(chain.invoke("Trường Đại học Công nghệ thông tin thành lập năm nào?"))
