# Import necessary libraries
import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from dotenv import load_dotenv
from typing import Optional, Dict, List

# Load environment variables from .env file and Streamlit secrets
load_dotenv()

# 🔐 Security Check - Verify all required API keys
def verify_api_keys():
    required_keys = {
        'GROQ_API_KEY': os.getenv("GROQ_API_KEY"),
        'OPENAI_API_KEY': os.getenv("OPENAI_API_KEY"),
        'LANGCHAIN_API_KEY': os.getenv("LANGCHAIN_API_KEY"),
        'HF_TOKEN': os.getenv("HF_TOKEN")
    }
    
    missing_keys = [name for name, value in required_keys.items() if not value]
    if missing_keys:
        st.error(f"❌ Missing API keys: {', '.join(missing_keys)}")
        st.markdown("Please add them to your `.streamlit/secrets.toml` file")
        st.stop()
    return required_keys

# 🔹 Constants Configuration
DEFAULT_MODEL = "deepseek-r1-distill-llama-70b"
MAX_CHAT_HISTORY = 20
MAX_SEARCH_RESULTS = 3
MAX_CONTENT_LENGTH = 500

# 🔹 Initialize API Wrappers with Configuration
def initialize_search_tools(max_results: int = MAX_SEARCH_RESULTS, 
                          max_content: int = MAX_CONTENT_LENGTH) -> list:
    arxiv_wrapper = ArxivAPIWrapper(
        top_k_results=max_results,
        doc_content_chars_max=max_content,
        load_max_docs=max_results
    )
    
    wiki_wrapper = WikipediaAPIWrapper(
        top_k_results=max_results,
        doc_content_chars_max=max_content
    )
    
    return [
        DuckDuckGoSearchRun(name="Web Search"),
        ArxivQueryRun(api_wrapper=arxiv_wrapper, name="Academic Papers"),
        WikipediaQueryRun(api_wrapper=wiki_wrapper, name="Wikipedia")
    ]

# 🔹 Session State Management
def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi! I'm an AI assistant with web search capabilities. How can I help you today?"}
        ]
    
    # Verify API keys are loaded
    st.session_state.api_keys = verify_api_keys()

# 🔹 Streamlit UI Configuration
def setup_ui():
    st.set_page_config(
        page_title="AI Search Assistant", 
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 AI Search Assistant")
    st.markdown("""
        **Powered by Groq, LangChain & Streamlit**  
        This assistant can search the web, academic papers, and Wikipedia to answer your questions.
    """)
    
    with st.sidebar:
        st.header("Settings")
        
        # Display API key status
        st.success("✅ All API keys configured")
        
        # Model selection
        st.session_state.model_name = st.selectbox(
            "Model",
            ["deepseek-r1-distill-llama-70b", "Llama3-70b-8192"],
            index=0
        )
        
        # Search configuration
        st.session_state.max_results = st.slider(
            "Max Results per Source",
            1, 5, MAX_SEARCH_RESULTS
        )
        
        if st.button("Clear Chat History"):
            st.session_state.messages = [
                {"role": "assistant", "content": "Chat history cleared. How can I help you now?"}
            ]
            st.rerun()

# 🔹 Chat Display Management
def display_chat_history():
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if "sources" in msg:
                with st.expander("Sources"):
                    st.json(msg["sources"])

# 🔹 Agent Initialization
def create_search_agent(llm: ChatGroq, tools: list) -> object:
    return initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
        verbose=True,
        max_iterations=10,
        early_stopping_method="generate"
    )

# 🔹 Main Application Logic
def main():
    initialize_session_state()
    setup_ui()
    display_chat_history()
    
    if prompt := st.chat_input("Ask me anything..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        try:
            # Initialize LLM with Groq
            llm = ChatGroq(
                groq_api_key=st.session_state.api_keys['GROQ_API_KEY'],
                model_name=st.session_state.model_name,
                temperature=0.3,
                streaming=True
            )
            
            # Initialize search tools
            tools = initialize_search_tools(
                max_results=st.session_state.max_results,
                max_content=MAX_CONTENT_LENGTH
            )
            
            # Create search agent
            search_agent = create_search_agent(llm, tools)
            
            # System message for structured responses
            system_message = f"""
            You are an advanced AI research assistant with access to multiple search tools.
            Current configuration:
            - Model: {st.session_state.model_name}
            - Max results per source: {st.session_state.max_results}
            """
            
            # Generate response
            with st.chat_message("assistant"):
                st_cb = StreamlitCallbackHandler(
                    st.container(),
                    expand_new_thoughts=True,
                    collapse_completed_thoughts=True
                )
                
                response = search_agent.run(system_message + "\n\nUser query: " + prompt, callbacks=[st_cb])
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response
                })
                st.write(response)
                
                if len(st.session_state.messages) > MAX_CHAT_HISTORY:
                    st.session_state.messages = st.session_state.messages[-MAX_CHAT_HISTORY:]
                
        except Exception as e:
            st.error(f"⚠️ An error occurred: {str(e)}")
            st.session_state.messages.append({
                "role": "assistant", 
                "content": "Sorry, I encountered an error processing your request. Please try again."
            })

if __name__ == "__main__":
    main()