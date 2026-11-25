from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_documents(docs: List[Document], chunk_size:int = 500, chunk_overlap:int = 100)  -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size , chunk_overlap = chunk_overlap)
    split_docs = []

    for doc in docs:
        texts = splitter.split_text(doc.page_content)
        #add metadata for the particular chunk
        for i,t in enumerate(texts):
            # convert metadata(info of the data --size,length,source) to dict 
            meta = dict(doc.metadata) if doc.metadata else {}
            #add chunk index to metadata
            meta.update({"chunk_index" : i})
            #add both chunk and metadata to splitdocs
            split_docs.append(Document(page_content= t ,metadata = meta))
    return split_docs
