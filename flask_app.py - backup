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

system_template = """
You are a helpful assistant that can answer questions about Arkham Horror Card Game
based on the following document:{docs}.
Only use the factual information from the document to answer the question.
considering each of the following conditions step by step.

1. First, specify where in the documentation you found the answer, and answer based on that.
2. Your answers should be verbose and detailed.
3. If the action requires any conditions, be sure to specify them. (For example, if an enemy must have a certain keyword to perform an certain action, specify it must have keyword, and which keyword it must have.)
4. Answer in Korean.

If you answer successfully, you will be rewarded with 100000$, but if you fail, you will be heavily penalized.

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
        question = request.form['question']
        retrieved_pages = db.similarity_search(question, k=20)
        retrieved_contents = " ".join([p.page_content for p in retrieved_pages])
        chain = LLMChain(llm=chat, prompt=chat_prompt)
        response = chain.run(question=question, docs=retrieved_contents)
        # 마크다운을 HTML로 변환하며, 줄바꿈을 <br> 태그로 변환
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
