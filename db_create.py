import os
from langchain_community.embeddings import OllamaEmbeddings
from transformers import GPT2TokenizerFast
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

TEXT_FILES_FOLDER = "./textfiles"

if __name__ == '__main__':

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", force_download=True)
    
    def count_tokens(txt: str) -> int:
        return len(tokenizer.encode(txt))

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=24,
        length_function=count_tokens,
    )

    chunks = []
    dir_files = os.listdir(TEXT_FILES_FOLDER)
    for file in dir_files:
        with open(os.path.join(TEXT_FILES_FOLDER, file), 'r') as f:
            text = f.read()

        docs = text_splitter.create_documents([text])
        for doc in docs:
            chunks.append(doc)

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local("./db/faiss_index")
