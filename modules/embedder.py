import os
from typing import List,Optional
#from langchain_community.vectorstores import FAISS,Pinecone as LC_Pinecone
from langchain_community.vectorstores import FAISS
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from modules.utils import get_embedding_model
import time
from pinecone import Pinecone,ServerlessSpec

#This method is going to build or update a vector store
def build_or_update_vectorstore(
        docs: List[Document],
        config:dict,
        #save the vector store
        persist_dir : Optional[str] = None
):
    # we get the provider from config file..in case if there is no value, make it default to openai and change the case
    provider = config.get("embedding_provider","openai").lower()

    #we get the vector store from config..
    store_config = config.get("vector_store",{})
    #now with store_config we can get all the values of vectorstore like type,local
    #if type not there , make it default to "faiss"
    store_type = store_config.get("type","faiss").lower()

    print(f"Embedding Provider: {provider}")
    print(f"Vector store type : {store_type}")

    #Load the embedding model
    #This method is written in utils.py
    embedding_model = get_embedding_model(provider)
    print("Embedding model loaded successfully..")
    

    #define local vector store
    #if persist dir does not exist
    if store_type in ['faiss']:
        if not persist_dir:
            persist_dir = os.path.join(os.getcwd(),"vectorstores",f"{provider}_{store_type}_index")
        os.makedirs(persist_dir,exist_ok=True)
        print("Index folder:", persist_dir)
        if store_type == "faiss":
            #index.faiss is the name of the index, where the index of the embedding is saved
            #this is saved as index.faiss in"faiss_path": "vectorstores/faiss_index" [from config]
            index_path = os.path.join(persist_dir,"index.faiss")
            #save the metada; .pkl is the extension for metadata file
            metadata_path = os.path.join(persist_dir,"index.pkl")

            if os.path.exists(index_path):
                #if data exists already we dont overwrite but we update
                #allow_dangerous_deserialization so that the save is not blocked
                print("Updating existing FAISS index..")
                vector_store = FAISS.load_local(persist_dir,embedding_model,allow_dangerous_deserialization=True)

                if docs:
                    #Internally langchain takes care everything embedding vector storage
                    vector_store.add_documents(docs)
                else:
                    print("No new documens provided - loaded existing index only")
            #there can be a need to build vectorstore from scratch
            else:
                if not docs:
                    print("No documents and no saved index yet")
                    return None
                    #raise ValueError("No documents provided and FAISS index not found")
                print("Creating new FAISS index..")
                vector_store = FAISS.from_documents(docs,embedding_model)

            vector_store.save_local(persist_dir)
            print(f"FAISS vectorstore saved at : {persist_dir}")
            return vector_store

    #============================================================
    # CLOUD VECTOR STORE (PINECONE)
    #============================================================

    elif store_type == "pinecone":
        print("💨 Using Pinecone vector store..")

        #--Load API credentials---
        api_key = os.getenv("PINECONE_API_KEY")
        #The environement line is not needed. It is never used afterwards.
        #  Current Pinecone does not need an environment name, and your config value gcp-starter is from the old version of Pinecone. You can delete the line.
        #environment = os.getenv("PINECONE_ENV") or store_config.get("cloud",{}).get("environment","us-east-1")
        if not api_key:
            raise ValueError("Missing PINECONE_API_KEY in environment variable")

        pc = Pinecone(api_key=api_key)
        #Look for the key cloud. If it’s there, return that section. If not, return an empty dictionary {}
        #In whatever came back, look for index_name. If it’s there, use it. If not, use the default
        #Returning {} means the second step still works, finds nothing, and falls back to the default.
        index_name = store_config.get("cloud",{}).get("index_name","rag-assistant-index")
        namespace = store_config.get("cloud",{}).get("namespace","default")

        """ 
        This block measures how many numbers one embedding has. Pinecone needs that number when it creates the index, so the code finds it by embedding a test sentence and counting.
        Why the number is needed
        Every embedding model turns text into a list of numbers, and each model always gives the same length. 
        OpenAI’s text-embedding-3-small gives 1536. 
        Pinecone’s index must be created with exactly that size, or it rejects your vectors.
        The sample doesn’t have to be a sentence. The model embeds whatever text you give it. 
        In your project that is a chunk from the splitter, which may be several sentences or part of a paragraph. This is why chunk size matters.
        One vector per chunk, not per word. If a chunk has 100 tokens, you still get one vector, not 100. 
        The model combines the meaning of all the tokens into that single list.
                """
        #--Detect embedding model dimension dynamically---
        print("Detecting embedding dimension...")
        test_text = ["Hello worl!d!"]

        try:
            #embed_documents always returns a list of embeddings, because it can handle many sentences at once.
            # With one sentence you still get a list, with one embedding inside. [0] takes that embedding out of the list so you can count its numbers.
            #test_text = [ "Hello", "Good morning" ]
           
        #   [0] picks embedding_1, the one for "Hello"
        #   [1] would pick embedding_2, the one for "Good morning"
            
            #[0]  means  the position of the sentence, counting from 0. It does not mean a word position.
            sample_vec = embedding_model.embed_documents(test_text)[0]
            embedding_dim = len(sample_vec)
        except Exception :
            #fallback for some embedding APIs that use .embed query()
            #embed query() takes string not list
            embedding_dim = len(embedding_model.embed_query("Hello world!"))
        print(f"Embedding dimension detected : {embedding_dim}")

        #--List all exising indexes---
        existing_indexes = [ i["name"] for i in pc.list_indexes()]
        #we are trying to build an index with a name rag-assistant-index (listed in config)in our pine cone account
        #index is bascially your vector store where all our embeddings are saved
        #Dimension is the length of each vector, meaning how many numbers represent one chunk’s meaning, for example 1536.
        #It depends on model
        # if index_name not in existing_indexes:
        #     #Create a new index with detected dimension
        #     print(f"Creating pinecode index '{index_name}' with dim ={embedding_dim}")

            #this code is given by pinecone to decide whether it runs locally or on the server
            #Auto-detect serverless vs classic environment
            # if environment.endswith("-free") or environment.endswith("-starter"):
            #     from pinecone import PodSpec
            #     pc.create_index(
            #         name = index_name,
            #         dimension=embedding_dim,
            #         metric="cosine",
            #         spec=PodSpec(environment=environment)
            #     )
            # else:
        #         pc.create_index(
        #             name = index_name,
        #             dimension=embedding_dim,
        #             metric="cosine",
        #             spec = ServerlessSpec(cloud="aws",region=environment)

        #         )
        # else:
        #     #Verify dimension matches the existing index
        #     desc = pc.describe_index(index_name)
        #     server_dim = desc["dimension"]
        #     if server_dim != embedding_dim:
        #         raise ValueError(
        #             f"Dimension mismatch: Embedding model = {embedding_dim},"
        #             f"Pinecone index = {server_dim}""
        #             f"Please use a different index name or matching embedding provider"
        #         )
        # ---- Step: make sure the Pinecone index exists ----
        # An index is the "cupboard" in the cloud where all chunk vectors are stored.
        # Its dimension (length of each vector) must match the embedding model,
        # so we use the value detected earlier (embedding_dim).
        # Index not found in the Pinecone account, so create a new one
        if index_name not in existing_indexes:
            print(f"Creating Pinecone index '{index_name}' with dim = {embedding_dim}")
            pc.create_index(
                name=index_name,
                dimension=embedding_dim,# size of each vector, taken from the embedding model
                metric="cosine",# compare vectors by direction, which suits text meaning
                spec=ServerlessSpec(cloud="aws", region="us-east-1"), #free plansupports serverless on AWS us-east-1
            )
    # wait until the new index is ready
             # A new index takes a few seconds to start, so wait until it is ready
            while not pc.describe_index(index_name).status["ready"]:
                 time.sleep(1)
        else:
            # Index already exists, so check that its size fits our embedding model
            server_dim = pc.describe_index(index_name).dimension
            if server_dim != embedding_dim:
                ## A mismatch means a different embedding model was used earlier.
                # Stop with a clear message instead of failing later during upload.
                raise ValueError(
                    f"Dimension mismatch: embedding model = {embedding_dim}, "
                    f"Pinecone index = {server_dim}. "
                    f"Use a different index name or a matching embedding model."
                )
        # this method turns your chunks into vectors and uploads them to Pinecone
        #LC_Pinecone comes from langchain_community, which is the older Pinecone integration. 
        # It may not work properly with the current pinecone package you are using for login. 
        # The current class is PineconeVectorStore from the package langchain-pinecone. Its from_documents and its connect-to-existing option work the same way.
        """  
        docs (your chunks)
   │
   ▼  for each chunk:
   │     1. send its text to the embedding model  →  vector (1536 numbers)
   │     2. upload to Pinecone: vector + original text + metadata (file name, page)
   ▼
        index "rag-assistant-index", namespace "default"
        │
        ▼
        returns vector_store  ← a connection to that index"""
        # vector_store = LC_Pinecone.from_documents(
        #     #The chunks from your splitter
        #     docs,
        #     #The model that turns each chunk into a vector. It must be the same model used for the questions
        #     embedding=embedding_model,
        #     #Which cupboard to use
        #     index_name=index_name,
        #     #Which shelf inside the cupboard. Searches look only on that shelf
        #     namespace= namespace
        # )

        # print(f"Pinecone index '{index_name}' ready (namespace : {namespace},dim ={embedding_dim})")
        # return vector_store

        if docs:
            # upload time: embed the chunks and store them in Pinecone
            vector_store = PineconeVectorStore.from_documents(
                docs,
                embedding=embedding_model,
                index_name=index_name,
                namespace=namespace,
            )
            print(f"Added {len(docs)} chunks to Pinecone index '{index_name}' (namespace: {namespace})")
        else:
            # page load: nothing to upload, just connect to the existing index
            vector_store = PineconeVectorStore(
                index_name=index_name,
                embedding=embedding_model,
                namespace=namespace,
            )
            print(f"Connected to Pinecone index '{index_name}' (namespace: {namespace})")

        return vector_store