"""
modules/memory.py
-----------------
Handles session-based conversational memory for the RAG Chatbot.
Supports hybrid memory: summary memory + recent short-term context window.
"""


from langchain.schema.runnable import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationSummaryMemory
from langchain_core.chat_history import InMemoryChatMessageHistory


#stores or placeholders

SESSION_MESSAGE_HISTORY = {}
SESSION_SUMMARY_MEMORY = {}
SESSION_RECENT_MEMORY = {}

def get_sesion_message_history(session_id:str):
    if session_id not in SESSION_MESSAGE_HISTORY:
        SESSION_MESSAGE_HISTORY[session_id] = InMemoryChatMessageHistory()
    return SESSION_MESSAGE_HISTORY[session_id]

def get_session_summary_memory(session_id:str): 
    if session_id not in SESSION_SUMMARY_MEMORY:
        SESSION_SUMMARY_MEMORY[session_id] = ConversationSummaryMemory(
            llm = ChatOpenAI(model = "gpt-4o-mini",temperature = 0),
            chat_memory=get_sesion_message_history(session_id),
            return_messages= True
        )
    #return SESSION_MESSAGE_HISTORY[session_id]
    return SESSION_SUMMARY_MEMORY[session_id] 

# --- Public accessors ---
#return the entire summary of the session
"""
#Vikas code--begin
def get_memory_summary(session_id:str) -> str:
    memory = get_session_summary_memory(session_id)
    return memory.load_memory_variables({}).get("summary","")
#vikas code --end
"""

#Claude code begin # change summary to history
def get_memory_summary(session_id: str) -> str:
    memory = get_session_summary_memory(session_id)
    msgs = memory.load_memory_variables({}).get("history", [])
    return "\n".join(m.content for m in msgs)
#Claude code end

"""
this func gets last n messages
"""
# --- Hybrid memory: short-term context window ---
def get_recent_memory(session_id:str,n:int = 6):
    """Get last n Q&A pairs from session memory."""
    history = SESSION_RECENT_MEMORY.get(session_id,[])
    return history[-n:] if len(history) > n else history

"""
history will have my message and chatbot reply
get_recent_memory will return all messages..but this will not be formatted under a category
who sent what,so we are formatting it
so itereate each item in history(turn)


"""
def get_recent_memory_formatted(session_id:str,n:int =6)->str:
    """Format recent memory as readable chat text."""
    history = get_recent_memory(session_id,n)
    if not history:
        return "No recent chat history available for this session"
    formatted = []

    for turn in history:
        formatted.append(f"User : {turn['user']} \nAssistant {turn['ai']}")
    return "\n\n".join(formatted)


#return last N conversations
def get_recent_context(session_id:str , n:int = 6)-> str:
    return get_recent_memory_formatted(session_id,n)

def update_recent_memory(session_id : str,user_input:str,ai_output:str,n:int = 6):
    """Append Q&A turn and maintain rolling window."""
    if session_id not in SESSION_RECENT_MEMORY:
        SESSION_RECENT_MEMORY[session_id] = []
    SESSION_RECENT_MEMORY[session_id].append({"user":user_input,"ai":ai_output})
    SESSION_RECENT_MEMORY[session_id] = SESSION_RECENT_MEMORY[session_id][-n:]


#Integrate memory into rag chain
def add_memory_to_chain(rag_chain,session_id,enabled: bool = True,recent_n:int = 6):
    if not enabled:
        print("Memory OFF - stateless chat")
        return rag_chain
    print(f"Hybrid Memory ON for session: { session_id}")
    summary_memory = get_session_summary_memory(session_id)

    def chain_with_summary(input_data , config = None):
        response = rag_chain.invoke(input_data,config=config)
        user_input = input_data["question"]
        ai_output = getattr(response,"content",str(response))
        #save to summary memory
        summary_memory.save_context({"input":user_input},{"output" : ai_output})

         #save to recent short-term memory
        update_recent_memory(session_id,user_input,ai_output,n=recent_n)

        #refresh summary/memory

        messages = summary_memory.chat_memory.messages
        existing_summary = summary_memory.load_memory_variables({}).get("summary","")
        new_summary = summary_memory.predict_new_summary(messages,existing_summary)
        summary_memory._buffer = new_summary # safe internal use
        print(f"🧠 Updated summary (first 250 chars):\n{new_summary[:250]}...\n")
        # print(f"🕓 Recent {recent_n} Q&A turns stored.")


        return response

    runnable_with_summary = RunnableLambda(chain_with_summary)

    return RunnableWithMessageHistory(
        runnable_with_summary,
        get_session_history= lambda _: get_sesion_message_history(session_id),
        input_messages_key="question",
        history_messages_key="history",
        output_messages_key=None
    )
