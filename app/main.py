import streamlit as st
from app.react_agent import VecStore, RetrieverTools, ChatVecstore, AgentMaker, ContextBuilder, llms

st.title("PhBot: Your Philosophical AI Assistant")

if "agent_maker" not in st.session_state:
    vecdb = VecStore()
    vecstore = vecdb.create_get_collection()

    models = llms()

    chatdb = ChatVecstore()
    context_builder = ContextBuilder(chat_vecstore=chatdb, models=models)

    retriever_tools = RetrieverTools(vecstore=vecstore, query_generator_llm=models["query_generator"])

    agent_maker = AgentMaker(retriever_tools=retriever_tools, models=models)
    agent_maker.agent_tools()
    agent_maker.first_response_chain()
    executor = agent_maker.agent()

    st.session_state["agent_maker"] = agent_maker
    st.session_state["executor"] = executor
    st.session_state["context_builder"] = context_builder
    st.session_state["chatdb"] = chatdb
    st.session_state["messages"] = []

# chat history # TODO: Implement chat history and test
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# user input
if prompt := st.chat_input("Ask me anything about philosophy!"):
    # show user message
    user_interaction = {"role": "user", "content": prompt}
    st.session_state.messages.append(user_interaction)
    with st.chat_message("user"):
        st.markdown(prompt)

    agent_maker = st.session_state["agent_maker"]
    executor = st.session_state["executor"]
    context_builder = st.session_state["context_builder"]
    chatdb = st.session_state["chatdb"]

    # agent router
    rsp = agent_maker.router(query=prompt, executor=executor, context_builder=context_builder)

    assistant_response = {"role": "assistant", "content": rsp}
    st.session_state.messages.append(assistant_response)
    with st.chat_message("assistant"):
        st.markdown(rsp)

    buffer = chatdb.add_conversation_turn(user_interaction, assistant_response)