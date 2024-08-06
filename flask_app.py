from flask import Flask, request, render_template
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
client = openai.OpenAI(
    # OPEN API KEY 설정
    api_key=os.environ.get("OPENAI_API_KEY"),
)

app = Flask(__name__)
socketio = SocketIO(app)  # Create a SocketIO instance

# Embeddings 및 FAISS 로드
embeddings = OpenAIEmbeddings()
db = FAISS.load_local('./faiss', embeddings, allow_dangerous_deserialization=True)

# ChatGPT 모델 및 프롬프트 설정
chat = ChatOpenAI(model_name="gpt-4o", temperature=0.5)

system_template = """
You are a helpful assistant that can answer questions about Arkham Horror Card Game
based on the following document:{docs}.
Only use the factual information from the document to answer the question.
considering each of the following conditions step by step.
1. If my question is Korean, Translate it into English first, and you must think as English.
2. Your answers should be verbose and detailed.
3. If the action requires any conditions, be sure to specify them. (For example, if an enemy must have a certain keyword to perform a certain action, specify it must have a keyword, and which keyword it must have.)
4. Translate your answer into Korean.
5. Don't print English answer.
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
    if request.method == 'POST':
        question = request.form['question']
        retrieved_pages = db.similarity_search(question, k=20)
        retrieved_contents = " ".join([p.page_content for p in retrieved_pages])
        chain = LLMChain(llm=chat, prompt=chat_prompt)
        response = chain.run(question=question, docs=retrieved_contents)
        return render_template('index.html', response=response)
    return render_template('index.html', response='')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    socketio.run(app, host='0.0.0.0', port=port, allow_unsafe_werkzeug=True)
    