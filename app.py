from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms.ollama import Ollama
from langchain_community.vectorstores import FAISS
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from flask import Flask, render_template, request, session
import time
import uuid

app = Flask(__name__)
app.config['SECRET_KEY'] = 'df0331cefc6c2b9a5d0208a726a5d1c0fd37324feba25506'

embeddings = OllamaEmbeddings(model="nomic-embed-text")
db = FAISS.load_local("./db/faiss_index", embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever()
llm = Ollama(model="llama3")

contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Format output for best readability on console screen. \

{context}"""

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

CHAT_SESSION_ID_KEY = "chat_session_id"
CHAT_SESSION_TEXT_KEY = "chat_session_text"


@app.route('/', methods=('GET', 'POST'))
def index():
    response_time = ""
    chat_session_text = ""
    answer = ""

    if request.method == 'POST':
        query = request.form['query']

        if query is not None:
            chat_session_id = uuid.uuid4()
            if CHAT_SESSION_ID_KEY in session:
                chat_session_id = session.get(CHAT_SESSION_ID_KEY)
                chat_session_text = session.get(CHAT_SESSION_TEXT_KEY)
            else:
                session[CHAT_SESSION_ID_KEY] = chat_session_id
            chat_session_text += query
            chat_session_text += "\n\n"
            tic = time.perf_counter()
            answer = conversational_rag_chain.invoke(
                input={"input": query},
                config={"configurable": {"session_id": f'{chat_session_id}'}}
            )["answer"]
            toc = time.perf_counter()
            response_time = f"in {(toc - tic):0.4f} seconds"
            chat_session_text += answer
            chat_session_text += "\n\n"
            session[CHAT_SESSION_TEXT_KEY] = chat_session_text

            print(answer)

    return render_template('index.html', answer=answer, response_time=response_time)
