import sys,os
import streamlit as st
#uuid - random id generation
import json,uuid 

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
#abspath -D:\Generative AI Dev Projects\RAG_Capstone_Project\app\streamlit_app.py
#relativepath = app\streamlit_app.py
#__file__ =D:\Generative AI Dev Projects\RAG_Capstone_Project\app\streamlit_app.py
#os.path.dirname(__file__) = D:\Generative AI Dev Projects\RAG_Capstone_Project\app
#join ".." and abspath	D:\Generative AI Dev Projects\RAG_Capstone_Project

print(f"Project root is {project_root}")
#print(sys.path)
#What sys.path is

#sys.path is a list of folders. When you write import something, Python checks these folders one by one, from first to last, until it finds it.
sys.path.insert(0,project_root)
#print(sys.path[0])

from modules.loader import load_document
from modules.splitter import split_documents
from modules.embedder import build_or_update_vectorstore
from modules.retriever import get_retriever
from modules.rag_chain import build_rag_chain
from modules.memory import add_memory_to_chain

#define page config

#define page config
st.set_page_config(page_title="Capstone RAG Chatbot",layout="wide")

#load_config
CONFIG_PATH = os.path.join(project_root,"config","config.json")
with open(CONFIG_PATH,"r") as f:
    config = json.load(f)

#session and muiltichat management
#There may be many sessions but only 1 will be active
#st.session_state is a special storage box that survives reruns.
# chat_sessions is a dictionary
#{} is an empty dictionary. Each chat is stored in it as key → value.
# """
# st.session_state
#    ├── st.session_state["chat_sessions"]   ← bracket style (dictionary-like)
#    └── st.session_state.chat_sessions      ← dot style (object-like)
#           │
#           └── both reach the same stored item
#           """

if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {}

#active session is only one

if "active_session" not in st.session_state:
    st.session_state.active_session = None
    


#sidebar : chat sessions

st.sidebar.header("💬Chat Settings") 

# new chat button
#click button to choose a new chat

def createchat() :
   
    new_id = uuid.uuid4()
    #new id is a key of the dict chat_seesions
    st.session_state.chat_sessions[new_id]= {
        "name":  f"Chat{len(st.session_state.chat_sessions) +1} ",
        "llm_provider" : "openai",        
        "memory_enabled" : True,
        "chat_history":[]
        
    }
    #active_session = "a3f1"          ← just the id       │
    #chat_sessions["a3f1"]            ← look up the details whenever needed
    st.session_state.active_session  = new_id

# by default, initialize first chat if there are no chat sessions
# """
# if not st.session_state.chat_sessions:
#     new_id = uuid.uuid4()
#     st.session_state.chat_Sessions[new_id] = {
#         "name" : f"Chat1",
#         "llm_provider" : "openai",
#         "memory_enabled" : True,
#         "chat_history" : []
#     }
#     """
if not st.session_state.chat_sessions:
    createchat()

if st.sidebar.button("➕New Chat"):
    createchat()

#toggle between the chats listed in dropdown

# """
# for sid,info in st.session_state.chat_sessions.items():
#     session_names[sid] = info["name"]

# """

session_names = {
    sid : info["name"] for sid,info in st.session_state.chat_sessions.items()
}


#selected box like dropdownlist
# """ 
# active_id = st.session_state.active_session                      # hop 1: which chat is open
# active_name = st.session_state.chat_sessions[active_id]["name"]  # hop 2: its name
# names = list(session_names.values())                             # all names, as a list
# active_index = names.index(active_name)     


# session_names = {"a3f1": "Chat 1", "b7c2": "Chat 2"}

# session_names.values()
# # dict_values(['Chat 1', 'Chat 2'])    ← NOT a list
# list(...) converts something that is not a list into a real list
# You need .index(). Only real lists have it:
# """
#alternate way
# ids = list(st.session_state.chat_sessions.keys())

# st.session_state.active_session = st.sidebar.selectbox(
#     "Active Chat Session",
#     options=ids,
#     index=ids.index(st.session_state.active_session),
#     format_func=lambda sid: st.session_state.chat_sessions[sid]["name"],   # ← here
# )
#format_func is a small function that Streamlit calls once
#  for each option. It takes an id and returns the text to displa

selected_item = st.sidebar.selectbox(
    "Active Chat Session",
    options = list(session_names.values()),
    index = list(session_names.values()).index(
            st.session_state.chat_sessions[st.session_state.active_session]["name"]),

)

#in backend  save the selected item in the activesession

for sid,info in st.session_state.chat_sessions.items():
    if info["name"] == selected_item:
        st.session_state.active_session = sid
        break

#current session

current_session = st.session_state.chat_sessions[st.session_state.active_session]

#optional - if you want to print the session_id in UI

st.sidebar.text(f"Session ID {st.session_state.active_session}")

#session - select specific id provider

current_session["llm_provider"] = st.sidebar.selectbox(
    "LLM_Provider",
    options=["openai","gemini"],
    index = 0 if current_session["llm_provider"]=="openai" else 1,
    key=f"provider_{st.session_state.active_session}",

)

# session - memory toggle
current_session["memory_enabled"] = st.sidebar.toggle(
    "Enable Memory",
    value= current_session["memory_enabled"],
    key = f"memory_{st.session_state.active_session}",
)

#Upload document section

st.sidebar.subheader("📁Upload Document")
uploaded_file = st.sidebar.file_uploader(
    "Upload PDF/DOC/TXT", type=["pdf","docx","docs","txt"]


)


#Vectorstore path
os.path.join(project_root,"vectorstores","faiss_index")

if uploaded_file:
    with st.spinner("Processing file document"):
        uploaded_path = os.path.join(project_root,"data","uploaded_docs",uploaded_file.name)
    os.makedirs(os.path.dirname(uploaded_path),exist_ok=True)
    #uploaded_file.getbuffer()	The file's raw contents, as bytes
    #f.write(...)	Writes the contents into the file
    #"w"	Write text	.txt files you build from strings
    #"wb"	Write bytes (binary)	PDFs, images, Word files, any upload
    # The uploaded file is a photocopy in your hand (in memory). If you leave, it is gone. open(..., "wb")
    # is opening an empty drawer in the cabinet (disk), write is putting the photocopy in, and 
    # with is closing the drawer for you when you finish, even if something goes wrong.
    with open(uploaded_path,"wb") as f:
        f.write(uploaded_file.getbuffer())


    docs = load_document(uploaded_path)
    chunks = split_documents(docs,chunk_size=400,chunk_overlap=50)
    vectorstore = build_or_update_vectorstore(chunks,config)

    st.success(f"{uploaded_file.name} processed and added to vector store")


#build rag chain
try:
    vectorstore = build_or_update_vectorstore([],config)
    if vectorstore is None:
        st.info("📄 No documents yet. Please upload a file in the sidebar to begin.")
        st.stop()

    retriever = get_retriever(vectorstore,config)
    config["llm_provider"] = current_session["llm_provider"]
    rag_chain = build_rag_chain(retriever,config)

    rag_chain_with_memory = add_memory_to_chain(
        rag_chain,
        session_id = st.session_state.active_session,
        enabled = current_session["memory_enabled"],
    )
except Exception as e:
    st.error(f"Error initializing backend: {e}")
    st.stop()


#Chat UI section
st.title("🤖Capstone RAG Chatbot")                          


# What st.chat_message does

# It is a Streamlit feature that draws a chat bubble. The only thing it needs from you is the speaker's name:

# "user" draws a bubble with a person avatar.
# "assistant" draws a bubble with a robot avatar.

#It does not store anything and does not create keys. It is purely a drawing instruction.
#display chat messages for current session
# chat_message = the envelope (who is speaking).
# with = put things inside the envelope.
# markdown = the letter written inside.


# Streamlit objects like st.chat_message(...) are containers: they can hold other Streamlit elements.
# The with statement says "everything indented below goes inside this container."
# st.markdown() displays text, and it understands Markdown, which is a simple way of formatting text with symbols:


for msg in current_session["chat_history"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# st.chat_input("Ask your question") draws the text box at the bottom of the page. Each time the script runs, it gives back:

# Situation	Returned value
# User has not submitted anything-- returns	None
# User typed "What is RAG?" and pressed Enter	-- returns "What is RAG?"
#:= is called the walrus operator (it looks like a walrus's eyes and tusks). It stores a value in a variable and uses that value in the same line.

# python
# prompt = st.chat_input("Ask your question")
# if prompt:

if prompt := st.chat_input("Ask your question"):
    current_session["chat_history"].append({"role" : "user","content" : prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        try:
            #rag_chain_with_memory is a LangChain object, not a plain function, so you can't call it with () directly. You have to call its .invoke() method.
            response = rag_chain_with_memory.invoke(
                {"question":prompt},
                config = {"configurable" : {"session_id" :  st.session_state.active_session}},
            )

            full_response = getattr(response,"content",str(response))
        except Exception as e:
            full_response = f"Error : {e}"

        response_placeholder.markdown(full_response)
    current_session["chat_history"].append({"role":"assistant","content":full_response})

            