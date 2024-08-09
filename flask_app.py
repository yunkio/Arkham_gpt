from flask import Flask, request, render_template, redirect, url_for, session
from flask_socketio import SocketIO, emit
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
import os
import openai
import bcrypt
import markdown

client = openai.OpenAI(
    # OPEN API KEY 설정
    api_key=os.environ.get("OPENAI_API_KEY"),
)

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "default_secret_key")  # 세션을 위한 비밀키 설정
socketio = SocketIO(app)  # Create a SocketIO instance

# 해시된 비밀번호 (미리 생성된 값을 사용)
hashed_password = b'$2b$12$PYwereRO4g.y0QMN6/wT9eADpjXAYNLAigARt7S7zFuwstaWyvzPG'

# Embeddings 및 FAISS 로드
embeddings = OpenAIEmbeddings()
db = FAISS.load_local('./faiss', embeddings, allow_dangerous_deserialization=True)

# ChatGPT 모델 및 프롬프트 설정
chat = ChatOpenAI(model_name="gpt-4o", temperature=0.4)

system_template="""
# your role
You are a brilliant expert at understanding the intent of the questioner and the crux of the question about 'Arkham Horror Card Game', and providing the most optimal answer to the questioner's needs from the documents you are given.

# Instruction
Your task is to answer the question using the following pieces of retrieved context delimited by XML tags.

<retrieved context>
Retrieved Context:
{context}
</retrieved context>

# Constraint
1. Think deeply and multiple times about the user's question\nUser's question:\n{question}\nYou must understand the intent of their question and provide the most appropriate answer.
- Ask yourself why to understand the context of the question and why the questioner asked it, and provide an appropriate response based on what you understand.
2. Choose the most relevant content(the key content that directly relates to the question) from the retrieved context and use it to generate an answer.
3. Keep the answer concise but logical/natural/in-depth.
4. 만약 조건에 특정 키워드가 필요한 경우라면 키워드가 필요한지 여부와, 어떤 키워드가 필요한지를 반드시 명시해줘. 키워드는 매우매우 중요하므로 이 규칙은 반드시 지켜야해. 예를 들면, 적이 어떤 행동을 하기 위해서는 특정 키워드가 필요한 경우, 어떤 키워드가 필요한 지에 대해 명시해줘.
5. 답변을 한국어로만 해.
"""
system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

human_template = "Answer the following question: {question}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages(
    [system_message_prompt, human_message_prompt]
)

# 홈 페이지
@app.route('/', methods=['GET', 'POST'])
def index():
    if not session.get('authenticated'):
        return redirect(url_for('login'))

    if request.method == 'POST':
        # query
        question = request.form['question']

        # retriever
        k = 5
        faiss_vectorstore = FAISS.from_documents(split_docs, OpenAIEmbeddings())
        retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": k})

        #rag_chain
        rag_chain = (
        {"context": itemgetter("context"), "question": itemgetter("question")}
        | prompt
        | llm
        | StrOutputParser()
        )
        #response
        retrieved_docs = retriever.get_relevant_documents(question)
        response = rag_chain.invoke({"context": retrieved_docs, "question": question})
        response_html = markdown.markdown(response, extensions=['nl2br'])
        return render_template('index.html', response=response_html)
    return render_template('index.html', response='')

# 로그인 페이지
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        password = request.form['password']
        if bcrypt.checkpw(password.encode(), hashed_password):
            session['authenticated'] = True
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error='Incorrect password.')
    return render_template('login.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    socketio.run(app, host='0.0.0.0', port=port, allow_unsafe_werkzeug=True)
