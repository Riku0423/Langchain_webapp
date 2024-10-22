import os
import streamlit as st

from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder


# .envファイルから環境変数を読み込む
load_dotenv()

st.title("LangcChain_streamlit_bot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


prompt = st.chat_input("Enter your prompt")
print(prompt)

def create_agent_chain():
    Chat = ChatOpenAI(
        model=os.getenv("OPENAI_API_MODEL"),
        temperature=float(os.getenv("OPENAI_API_TEMPERATURE", 0.7)),
        streaming=True
    )

    agent_kwargs = {
        "extra_prompt_messages":[MessagesPlaceholder(variable_name="memory")]
    }

    memory = ConversationBufferMemory(memory_key="memory",return_messages=True)


    tools = load_tools(["ddg-search","wikipedia"])
    return initialize_agent(
        tools, 
        Chat, 
        agent=AgentType.OPENAI_FUNCTIONS, 
        verbose=True,
        agent_kwargs=agent_kwargs,
        memory=memory
        )

if "agent_chain" not in st.session_state:
    st.session_state.agent_chain = create_agent_chain()

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(f"User: {prompt}")

    with st.chat_message("assistant"):
        callback = StreamlitCallbackHandler(st.container())
        agent_chain = st.session_state.agent_chain
        response = agent_chain.run(prompt, callbacks=[callback])
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
