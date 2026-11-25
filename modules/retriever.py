"""
Base vs Multiquery retriever

we ask a qusetion and the question is embedded and it is searched in vectorstore and the top documents 
are retrieved

doc is complex or deeply structured
when we use Multiquery retriever, it tries to understand our question from diff dimension
Instead of using a knowledge base, we use LLM which will rewrite the query in multiple styles
and create embedding for all of it and then based on that we do similarity search and we fetch the relevant data and we give it to LLM in the rag.

when to use what?

Employee policy - single retriever
multidimensional question - multiquery retriever(latency,cost will be higher)


"""

from langchain.retrievers import MultiQueryRetriever
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from modules.utils import get_embedding_model
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI


def get_retriever(vectorstore,config):
    #retriever_type = config.get("retriever_type","base").lower()
    retriever_type = config.get("retriever_type","multiquery").lower()    

    if retriever_type == "base":
        print("Using base retreiver(simple similarity search)")
        return vectorstore.as_retriever(search_kwargs = {"K":6})
    elif retriever_type == "multiquery":
        print("Using multi query retreiver...")
        llm_provider = config.get("llm_provider","openai").lower()
        if llm_provider == "openai":
            llm = ChatOpenAI(model = "gpt-4o-mini",temperature=0)
        elif llm_provider == "gemini":
            llm = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")
        else:
            raise ValueError(f"Unsupported LLM provider : {llm_provider}")

        return MultiQueryRetriever.from_llm(retriever =  vectorstore.as_retriever(search_kwargs = {"K":6}),llm = llm)
            # we pass the llm and query rewritting can be done
            
    else:
        raise ValueError(f"Unsupported retriever_type : {retriever_type}")


