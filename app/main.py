# ライブラリをインポート
import streamlit as st
from streamlit_chat import message
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)


#プロンプトテンプレートを作成
template = """
 あなたはおじさん構文で返答を行ってください。\
                  おじさん構文の特徴は以下の通りです。\
                  *絵文字や顔文字を多用する\
                  *語尾にカタカナを使う\
                  *句読点を付ける\
                  *長文で返す\
                  *聞かれてないのに自分の近況報告を行う\
                  *そこはかとなく下心が感じられる文章\
                  *親しくないのにメッセージになるとタメ口\
                  おじさん構文の例をあげます。\
                  お疲れさまです。というメッセージに対して，以下のような返答をします。\
                  お疲れサマ😃♥こんな遅い時間💤✋😎に何をしているのかな⁉️😍突然だけど、〇〇ちゃんは中華🍜好きカナ😜⁉️小生は明日から北京だよ😃😃✋テレビに写っちゃったらどうしよ〜(^o^)
"""

# 会話のテンプレートを作成
prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(template),
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("{input}"),
])

#会話の読み込みを行う関数を定義
@st.cache_resource
def load_conversation():
    llm = ChatOpenAI(
        streaming=True,
        model_name="gpt-3.5-turbo",
        temperature=0
        
    )
    memory = ConversationBufferMemory(return_messages=True)
    conversation = ConversationChain(
        memory=memory,
        prompt=prompt,
        llm=llm)
    return conversation

# 質問と回答を保存するための空のリストを作成
if "generated" not in st.session_state:
    st.session_state.generated = []
if "past" not in st.session_state:
    st.session_state.past = []

# 送信ボタンがクリックされた後の処理を行う関数を定義
def on_input_change():
    user_message = st.session_state.user_message
    conversation = load_conversation()
    answer = conversation.predict(input=user_message)

    st.session_state.generated.append(answer)
    st.session_state.past.append(user_message)
    st.session_state.user_message = ""

# タイトルやキャプション部分のUI
st.title("おじさん構文BOT")
st.caption("おじさん構文")
st.write("おじさんが質問に答えます。")

# 会話履歴を表示するためのスペースを確保
chat_placeholder = st.empty()

# 会話履歴を表示
with chat_placeholder.container():
    for i in range(len(st.session_state.generated)):
        message(st.session_state.past[i],is_user=True)
        message(st.session_state.generated[i])

# 質問入力欄と送信ボタンを設置
with st.container():
    user_message = st.text_area("質問を入力する", key="user_message")
    st.button("送信", on_click=on_input_change)