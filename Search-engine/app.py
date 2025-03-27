# Import necessary libraries
import os
import streamlit as st  # For building the web app interface
from langchain_groq import ChatGroq  # Groq's high-performance LLM
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper  # Academic research tools
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun  # Search tools
from langchain.agents import initialize_agent, AgentType  # LangChain agent setup
from langchain.callbacks import StreamlitCallbackHandler  # Handles AI thoughts in Streamlit UI
from dotenv import load_dotenv  # For loading environment variables
from typing import Optional, Dict, List  # For type hints

# Load environment variables from .env file
load_dotenv()

# 🔹 Constants Configuration
DEFAULT_MODEL = "deepseek-r1-distill-llama-70b"  # Default Groq model
MAX_CHAT_HISTORY = 20  # Maximum messages to retain in chat history
MAX_SEARCH_RESULTS = 3  # Default number of results for search tools
MAX_CONTENT_LENGTH = 500  # Max characters for content snippets

# 🔹 Initialize API Wrappers with Configuration
def initialize_search_tools(max_results: int = MAX_SEARCH_RESULTS, 
                          max_content: int = MAX_CONTENT_LENGTH) -> list:
    """
    Initialize and configure search tools with proper settings.
    
    Args:
        max_results: Number of results to return per tool
        max_content: Maximum content length for results
        
    Returns:
        List of configured search tools
    """
    # ArXiv API for academic papers
    arxiv_wrapper = ArxivAPIWrapper(
        top_k_results=max_results,
        doc_content_chars_max=max_content,
        load_max_docs=max_results
    )
    
    # Wikipedia API for general knowledge
    wiki_wrapper = WikipediaAPIWrapper(
        top_k_results=max_results,
        doc_content_chars_max=max_content
    )
    
    # Configured search tools
    return [
        DuckDuckGoSearchRun(name="Web Search"),  # General web search
        ArxivQueryRun(api_wrapper=arxiv_wrapper, name="Academic Papers"),  # Academic research
        WikipediaQueryRun(api_wrapper=wiki_wrapper, name="Wikipedia")  # Encyclopedia
    ]

# 🔹 Session State Management
def initialize_session_state() -> None:
    """
    Initialize or reset the Streamlit session state variables.
    """
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi! I'm an AI assistant with web search capabilities. How can I help you today?"}
        ]
    
    # Get API key only from environment variables
    st.session_state.groq_api_key = os.getenv("GROQ_API_KEY")

# 🔹 Streamlit UI Configuration
def setup_ui() -> None:
    """
    Configure the Streamlit user interface layout and elements.
    """
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
    
    # Sidebar for settings
    with st.sidebar:
        st.header("Settings")
        
        # Display API key status
        api_status = "✅ Configured" if st.session_state.groq_api_key else "❌ Missing"
        st.write(f"API Key Status: {api_status}")
        
        if not st.session_state.groq_api_key:
            st.error("Please set GROQ_API_KEY in your environment variables")
            st.markdown("[Get your API key](https://console.groq.com/keys)")
            st.stop()
        
        # Model selection
        st.session_state.model_name = st.selectbox(
            "Model",
            ["deepseek-r1-distill-llama-70b", "Llama3-70b-8192"],
            index=0,
            help="Select the Groq model to use"
        )
        
        # Search configuration
        st.session_state.max_results = st.slider(
            "Max Results per Source",
            1, 5, MAX_SEARCH_RESULTS,
            help="Number of results to fetch from each source"
        )
        
        if st.button("Clear Chat History"):
            st.session_state.messages = [
                {"role": "assistant", "content": "Chat history cleared. How can I help you now?"}
            ]
            st.rerun()

# 🔹 Chat Display Management
def display_chat_history() -> None:
    """
    Display the chat message history in the Streamlit UI.
    """
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if "sources" in msg:
                with st.expander("Sources"):
                    st.json(msg["sources"])

# 🔹 Agent Initialization
def create_search_agent(llm: ChatGroq, tools: list) -> object:
    """
    Create and configure a LangChain agent for search tasks.
    
    Args:
        llm: Initialized Groq language model
        tools: List of search tools to use
        
    Returns:
        Initialized LangChain agent
    """
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
def main() -> None:
    """
    Main function to run the Streamlit application.
    """
    # Initialize the application
    initialize_session_state()
    setup_ui()
    
    # Display chat history
    display_chat_history()
    
    # Process user input
    if prompt := st.chat_input("Ask me anything..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        try:
            # Initialize LLM with Groq
            llm = ChatGroq(
                groq_api_key=st.session_state.groq_api_key,
                model_name=st.session_state.model_name,
                temperature=0.3,  # Balance creativity and factuality
                streaming=True
            )
            
            # Initialize search tools with current settings
            tools = initialize_search_tools(
                max_results=st.session_state.max_results,
                max_content=MAX_CONTENT_LENGTH
            )
            
            # Create search agent
            search_agent = create_search_agent(llm, tools)
            
            # System message for structured responses
            system_message = f"""
            You are an advanced AI research assistant with access to multiple search tools.
            Follow these guidelines:
            1. Always use the Thought → Action → Observation pattern
            2. Provide well-structured, concise responses
            3. Cite sources when available
            4. If unsure, say you don't know rather than guessing
            5. For complex queries, break them down into smaller questions
            
            Current configuration:
            - Model: {st.session_state.model_name}
            - Max results per source: {st.session_state.max_results}
            
            User query: {prompt}
            """
            
            # Generate response
            with st.chat_message("assistant"):
                st_cb = StreamlitCallbackHandler(
                    st.container(),
                    expand_new_thoughts=True,
                    collapse_completed_thoughts=True
                )
                
                response = search_agent.run(system_message, callbacks=[st_cb])
                
                # Store and display response
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response
                })
                st.write(response)
                
                # Limit chat history size
                if len(st.session_state.messages) > MAX_CHAT_HISTORY:
                    st.session_state.messages = st.session_state.messages[-MAX_CHAT_HISTORY:]
                
        except Exception as e:
            st.error(f"⚠️ An error occurred: {str(e)}")
            st.session_state.messages.append({
                "role": "assistant", 
                "content": f"Sorry, I encountered an error processing your request. Please try again."
            })

# Run the application
if __name__ == "__main__":
    main()