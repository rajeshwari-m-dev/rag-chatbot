from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.runnable import RunnableLambda,RunnableParallel
from modules.memory import get_memory_summary,get_recent_context

def build_rag_chain(retriever,config):
    provider = config.get("llm_provider","openai").lower()
    temperature = config.get("temperature",0.3)

    print(f"Building RAG chain using provider: {provider}")

    # Provider specific model setup
    if provider == "openai":
        llm = ChatOpenAI(
            model = config.get("openai_model","gpt-4o-mini"),
            temperature= temperature,
            streaming=True # future-proof for streaming
        )

    elif provider == "gemini":
        llm = ChatGoogleGenerativeAI(
            model = config.get("gemini_model","gemini-3.8-flash"),
            temperature = temperature,
            streaming = True 
        )

    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")

    #system prompt 

    system_prompt = """
    You are a helpful AI assistant for answering questions about the uploaded documents.

    You have a summary of the past conversation and the most recent turns.
    Use them to understand follow-up questions (for example "they", "that", "explain more").

    Rules:
    - Facts about the documents must come ONLY from the Retrieved Context.
    - If the Retrieved Context does not contain the answer, say exactly:
    "I could not find this in the provided documents."
    - If the user asks about the conversation itself (for example "what did I ask
    before?" or "what was my first question?"), answer from the conversation
    summary and recent turns.
    - If the user greets you, introduces themselves, thanks you, or makes small talk,
    reply politely and ask them to ask a question about the uploaded documents.
    - Answer in 1 to 2 sentences. Do not add closing remarks.
    """
    

    #unified prompt structure
    prompt = ChatPromptTemplate.from_template("""
    
    System Prompt:
    {system_prompt}

    Conversation Summary Memory:
    {memory_summary}

    Recent Chat History(last few turns):
    {recent_memory}

    Retrieved Context:
    {context}

    User Question:
    {question}

    If the user asks to summarize or refer to earlier parts of the conversation,
    rely primarly on the chat history and memory summary.
    Otherwise,combibe retrieved context and memories to answer effectively
    """)

    #Combine retrieved documents

    def combine_docs(docs):
        if isinstance(docs,dict):
            docs = docs.get("content",[])
        if not docs:
            return "No relevant context found"
        return "\n\n".join( doc.page_content for doc in docs)

    """
    context is you ask the qn,u use the retriever and then combine the docs

    and tells prompt to use this context and then the llm
    
    in this context, we use retriever to bring the info, we use the qn to ask, system prompt and memory related info
    """
    #Build hybrid RAG +memory chain
    #load for the module
    rag_chain = (RunnableParallel(
        {
             #final retrieved doc is used as a context;in RAG archi along with the qn we also send the context
            "context":RunnableLambda(lambda x:  x["question"]) | retriever | RunnableLambda(combine_docs),
            "question":RunnableLambda(lambda x:  x["question"]),
            "system_prompt":RunnableLambda(lambda _:system_prompt),
            #this session id is not in config..but we create dynamically when we invoke chain
            "memory_summary":RunnableLambda(lambda x, config : get_memory_summary(config["configurable"]["session_id"])),
            "recent_memory" :RunnableLambda(lambda x, config : get_recent_context(config["configurable"]["session_id"])),
        }
    )) | prompt | llm

    print(f"Hybrid RAG + Memory chain built successffully usingg {provider.upper()}")

    return rag_chain