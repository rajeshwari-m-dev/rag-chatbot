import os
from typing import List
from langchain_core.documents import Document

#what ever the docs type be, we shud be able to load the doc
import pypdf
import docx
import requests
from bs4 import BeautifulSoup

# we need to create a uniform format around all type of data
#convert plaintext into standardized object with metadata as we
#  use files from diff sourcers(text,pdf,doc,url)
# sec arg is optional .if no arg sent, we say as "inline"
#it returns the document object
# in return page_content sets the main content of the document
#metadata adds the source as dict
def text_to_doc(text:str, source: str = "inline") -> Document:
    return Document(page_content = text,metadata = {"source" : source})


def load_txt(path:str) -> List[Document]:
   if not os.path.exists(path):
       raise FileNotFoundError(path)
   with open(path,"r", encoding = "utf-8",errors = "ignore") as f:
       text = f.read()
    #as we are calling this from notebook , there we do not have 
    #any data folder..so find the parent path and search for document
   return [text_to_doc(text,source = os.path.abspath(path))]
   

def load_pdf(path:str) -> List[Document]:
    #get it in reader obj
    reader = pypdf.PdfReader(path)
    #create an empty list and we add contents to this empty list
    docs = []
    #reader has so many pages. so enumerate it
    for i,page in enumerate(reader.pages):
        text = page.extract_text() or ""
        docs.append(text_to_doc(text,source = f"{os.path.abspath(path)}:page : {i+1}"))
    return docs


    

def load_docx(path:str) -> List[Document]:
    doc = docx.Document(path)
    paragraphs = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
    text = "\n\n".join(paragraphs)
    return [text_to_doc(text,source=os.path.abspath(path))]

#load data from a web page
def load_url(url:str) -> List[Document]:
    #timeout (for any server issues)
    """
    Sends an HTTP GET request to the given url using the requests library,
     and stores the response in resp. 
     timeout=15 means: if the server doesn't respond within 15 seconds, raise an exception instead of hanging forever
       — this protects you from a slow/unresponsive server freezing your program.
    """
    #resp = requests.get(url,timeout=15)
    #says success or error
    


    #Wikipedia blocks requests that identify themselves as the default python-requests client. 
    # It's like a security guard who won't let in anyone without a name badge.
    #

    headers = {"User-Agent": "Mozilla/5.0 (compatible; RAGLoader/1.0)"}
    resp = requests.get(url, headers=headers,timeout=15)  #→  "I am a browser-like client"  →  200 ✓
    """
    Checks the HTTP status code of the response.
      If it's an error code (like 404 Not Found or 500 Server Error),
        this raises an exception immediately and stops execution. 
        If the request was successful (status 200), it does nothing and execution continues.
    """
    resp.raise_for_status()

    """
    resp.text is the raw HTML of the page as a string.
      BeautifulSoup(...) parses that raw HTML string into a navigable tree structure,
        so you can search and extract elements from it easily. 
        "html.parser" tells BeautifulSoup which parser engine to use (Python's built-in one, as opposed to lxml or html5lib).
    """
    soup = BeautifulSoup(resp.text,"html.parser")

    """
    Searches the parsed HTML tree for all tags matching any of: paragraphs (<p>), headings (<h1>, <h2>, <h3>), and list items (<li>).
      Returns them as a list of tag objects, in the order they appear on the page. 
      This is essentially saying "grab the readable, meaningful text-bearing elements, skip things like <script>, <nav>, <footer>, etc."
    """
 
    elements = soup.find_all(["p","h1","h2","h3","li"])

    """
        # e.get_text(strip=True)        → element's visible text, whitespace trimmed
    # if e.get_text(strip=True)     → skip empty elements
    # e.get_text(separator=" ", strip=True)  → same, but spaces join nested text
    # [... for e in elements if ...]   → list of cleaned text, one per element
    # "\n\n".join([...])            → join all into one string, blank line between each

text = "\n\n".join([e.get_text(separator=" ", strip=True) for e in elements if e.get_text(strip=True)])
    """
    text = "\n\n".join([e.get_text(separator=" ",strip = True) for e in elements if e.get_text(strip=True)])
    return[text_to_doc(text,source=url)]


    

#mother func which calls the above methods

def load_document(path_or_url : str) -> List[Document]:
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        return load_url(path_or_url)
    if not os.path.exists(path_or_url):
        raise FileNotFoundError(F"File not found at :{path_or_url}")

    ext = os.path.splitext(path_or_url)[1].lower()

    if ext in [".txt","md"]:
        return load_txt(path_or_url)
    elif ext == ".pdf":
        return load_pdf(path_or_url)
    elif ext in [".docx",".docs",".doc"]:
        return load_docx(path_or_url)
    else:
        try:
            return load_txt(path_or_url)
        except Exception:
            raise ValueError(f"Unsupported file type : {ext}")