import streamlit as st
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_deepseek.chat_models import ChatDeepSeek
#from langchain.agents import Tool
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
from typing import TypedDict, Dict, Any, List
from langchain_core.messages import AIMessage, HumanMessage
from langchain.tools import tool # Import for @tool decorator

from typing import TypedDict, Dict, Any, List

from langchain_experimental.functional_agents.base import FunctionAgent

# --- Page Configuration ---
st.set_page_config(
    page_title="Personalized Financial Advisor (MultiAgent + RAG)",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Sidebar ---
with st.sidebar:
    st.image("Bank_Financial_Advisor_Favicon.ico")  # Replace with your logo file if you have one
    st.title("Personalized Financial Advisor (MultiAgent + RAG)")
    st.markdown("MultiAgent Chatbot with RAG for personalized financial advice based on bank product info.")
    st.markdown("---")
    st.markdown("## Agent-Based Architecture (LangGraph)")
    st.markdown("- **Personalization Agent:** Gathers user financial info.")
    st.markdown("- **Risk Assessment Agent:** Assesses risk tolerance.")
    st.markdown("- **Investment Advisor Agent:** Provides investment advice using RAG.")
    st.markdown("- **Budgeting Advisor Agent:** Suggests budgeting tips.")
    st.markdown("---")
    st.markdown("## RAG Details:")
    st.markdown(f"- **Vector Database:** ChromaDB (`chroma_db`)")
    st.markdown(f"- **Embedding Model:** `all-MiniLM-L6-v2` (HuggingFace)")
    st.markdown("- **Knowledge Source:** Comprehensive Bank Information Document")
    st.markdown("---")
    st.markdown("## LLM & Agents:")
    st.markdown("- **LLM:** DeepSeek LLM (for all Agents)")
    st.markdown("- **LangGraph:** Agent orchestration framework")
    st.markdown("---")
    st.markdown("## About")
    st.markdown("""
    This chatbot uses a LangGraph MultiAgent architecture with Retrieval Augmented Generation (RAG), built with:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [LangGraph](https://python.langchain.com/docs/langgraph)
    - [Hugging Face Sentence Transformers](https://huggingface.co/sentence-transformers)
    - [ChromaDB](https://www.trychroma.com/)
    - [DeepSeek LLM](https://deepseek.ai/)
    """)

# --- Initialize Global Variables and Functions ---

# Embedding Model and VectorDB
embedding_model_name = "all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
persist_directory = "chroma_db"

# Load Vector Database
def load_vector_db():
    """Loads the persisted ChromaDB vector database."""
    try:
        vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        print(f"ChromaDB loaded from: {persist_directory}")
        return vectordb
    except Exception as e:
        st.error(f"Error loading ChromaDB: {e}")
        st.stop()
        return None

# Load DeepSeek LLM
def load_deepseek_llm():
    """Loads the DeepSeek LLM using API key from Streamlit secrets."""
    deepseek_api_key = st.secrets["DEEPSEEK_API_KEY"]
    try:
        llm = ChatDeepSeek(
            deepseek_api_key=deepseek_api_key,
            model="deepseek-chat",  # Or "deepseek-coder"
            temperature=0.7,
            top_p=0.9,
            model_kwargs={"max_tokens": 1000},
        )
        print("DeepSeek LLM initialized.")
        return llm
    except Exception as e:
        st.error(f"Error initializing DeepSeek LLM: {e}")
        st.stop()
        return None

# Global VectorDB instance (loaded at startup)
vectordb_global = load_vector_db()
llm_global = load_deepseek_llm() # Load LLM globally as well for tool functions

# --- Tool Definitions (Re-integrated and adapted) ---

@tool ()
def get_user_risk_tolerance(user_info: Dict) -> str:
    """Returns the user's risk tolerance based on user information."""
    risk_tolerance = user_info.get("risk_tolerance", "Unknown")
    return f"User's risk tolerance is: {risk_tolerance}"

@tool ()
def get_user_financial_goals(user_info: Dict) -> str:
    """Returns the user's financial goals based on user information."""
    financial_goals = user_info.get("financial_goals", "Unknown")
    return f"User's financial goals are: {financial_goals}"

@tool ()
def provide_investment_advice_rag(risk_tolerance: str, financial_goals: str, query: str) -> str:
    """Provides investment advice based on risk tolerance, financial goals, AND document retrieval.

    Args:
        risk_tolerance (str): User's risk tolerance level.
        financial_goals (str): User's financial goals.
        query (str): User's original question to guide the RAG and advice generation.
    """
    if vectordb_global is None:
        return "Vector database not initialized."

    retrieved_info = perform_vector_search(query) # Use vector search tool to get relevant document info

    advice_prompt = f"""
    Based on the user's risk tolerance: '{risk_tolerance}', financial goals: '{financial_goals}', 
    and the following information retrieved from the bank product document: 
    --- Retrieved Information ---
    {retrieved_info}
    --- End Retrieved Information ---

    Provide personalized investment advice. Consider different asset classes and diversification strategies 
    as mentioned in the retrieved document excerpts. Keep the advice concise and actionable for an individual investor.
    """
    response = llm_global.invoke(advice_prompt) # Use globally loaded LLM
    return response.content


@tool ()
def suggest_budgeting_tips(income: float, expenses: float) -> str:
    """Suggests budgeting tips based on income and expenses."""
    prompt = f"""
    Based on the user's income: ${income} and expenses: ${expenses}, 
    suggest personalized budgeting tips to improve their financial health. 
    Focus on practical and easy-to-implement advice.
    """
    response = llm_global.invoke(prompt) # Use globally loaded LLM
    return response.content


def perform_vector_search(query: str) -> str: # Re-use vector search function from previous code
    """Tool: Performs a vector similarity search in ChromaDB and returns relevant document excerpts."""
    if vectordb_global is None:
        return "Vector database not initialized."
    try:
        results = vectordb_global.similarity_search(query, k=3) # Search top 3 relevant documents
        source_documents_content = "\n\n".join([doc.page_content for doc in results]) # Combine content
        return source_documents_content or "No relevant information found in the document."
    except Exception as e:
        return f"Error during vector search: {e}"


# Bundle tools for each agent (ToolController is not directly used with FunctionAgent in LangGraph as of now - tools are passed directly)
personalization_agent_tools = [get_user_risk_tolerance, get_user_financial_goals] # Personalization agent tools
risk_assessment_agent_tools = [get_user_risk_tolerance] # Risk assessment agent tools
investment_advisor_agent_tools = [provide_investment_advice_rag, get_user_financial_goals, get_user_risk_tolerance] # Investment advisor tools
budgeting_advisor_agent_tools = [suggest_budgeting_tips] # Budgeting advisor tools

# Define Graph State
class AgentState(TypedDict):
    """State type for the agent graph."""
    user_info: Dict[str, Any]
    messages: List[AIMessage | HumanMessage]
    agent_response: str # To hold the last agent response for display

# --- Agent Node Definitions (Re-integrated and adapted) ---

def personalization_agent_node(state: AgentState):
    """Agent to personalize interaction and gather user info."""
    agent = FunctionAgent(
        llm=llm_global, # Use globally loaded LLM
        tools=personalization_agent_tools, # Pass relevant tools
        system_message=SystemMessage(
            "You are a Personalization Agent. Your role is to understand the user's financial situation and goals. "
            "Start by asking general questions to gather information like risk tolerance, financial goals, income, and expenses. "
            "Once you have enough information, pass the control to the next agent."
        )
    )
    # ... (rest of the node function - similar to original personalized_financial_advisor_app.py)
    agent_action = agent.run(state['messages']) # Get agent's action/response
    return {"messages": [HumanMessage(content=agent_action)]} # In LangGraph, return dict to update state


def risk_assessment_agent_node(state: AgentState):
    """Agent to assess user's risk tolerance."""
    agent = FunctionAgent(
        llm=llm_global, # Use globally loaded LLM
        tools=risk_assessment_agent_tools, # Pass relevant tools
        system_message=SystemMessage(
            "You are a Risk Assessment Agent. Your role is to clearly and concisely determine the user's risk tolerance level. "
            "Use available tools to understand the user's stated risk tolerance. If not available, infer from other information. "
            "Once assessed, pass control to the next agent."
        )
    )
    agent_action = agent.run(state['messages'])
    return {"messages": [HumanMessage(content=agent_action)]}


def investment_advisor_agent_node(state: AgentState):
    """Agent to provide investment advice (now with RAG)."""
    agent = FunctionAgent(
        llm=llm_global, # Use globally loaded LLM
        tools=investment_advisor_agent_tools, # Investment agent now has RAG tool
        system_message=SystemMessage(
            "You are an Investment Advisor Agent. Your role is to provide personalized investment advice based on the user's risk tolerance and financial goals. "
            "You also have access to a tool to search a bank product document to provide more informed and relevant advice. "
            "Use available tools to get user risk tolerance, financial goals, and relevant document information. Then provide actionable and concise advice, referencing the document where appropriate."
        )
    )
    agent_action = agent.run(state['messages'])
    return {"messages": [HumanMessage(content=agent_action)]}


def budgeting_advisor_agent_node(state: AgentState):
    """Agent to provide budgeting advice."""
    agent = FunctionAgent(
        llm=llm_global, # Use globally loaded LLM
        tools=budgeting_advisor_agent_tools, # Pass budgeting tools
        system_message=SystemMessage(
            "You are a Budgeting Advisor Agent. Your role is to provide personalized budgeting advice based on the user's income and expenses. "
            "Use available tools to get this information and then suggest practical and easy-to-implement budgeting tips."
        )
    )
    agent_action = agent.run(state['messages'])
    return {"messages": [HumanMessage(content=agent_action)]}


def user_input_node(state: AgentState):
    """Node to get user input from Streamlit UI."""
    print(f"Inside user_input_node, type of state: {type(state)}")
    user_message = state['user_info'].get("current_message")
    if user_message:
        return {"messages": [HumanMessage(content=user_message)]} # In LangGraph, just return updated messages
    else:
        return {"messages": []} # No new messages


def agent_response_node(state: AgentState):
    """Node to get agent's last response for display."""
    last_message = state['messages'][-1] if state['messages'] else None
    if isinstance(last_message, HumanMessage): # Consider last human message as agent's response in this flow
        return {"agent_response": last_message.content} # Return agent's response content specifically
    return {"agent_response": ""} # No agent response


# Define Graph Workflow (LangGraph)
workflow = StateGraph(AgentState)

# User Input Node
workflow.add_node("user_input", user_input_node)

# Agent Nodes
workflow.add_node("personalization_agent", personalization_agent_node)
workflow.add_node("risk_assessment_agent", risk_assessment_agent_node)
workflow.add_node("investment_advisor_agent", investment_advisor_agent_node)
workflow.add_node("budgeting_advisor_agent", budgeting_advisor_agent_node)
workflow.add_node("display_agent_response", agent_response_node) # Agent response node for display

# Edges - Define the flow of agents
workflow.add_edge("user_input", "personalization_agent")
workflow.add_edge("personalization_agent", "risk_assessment_agent")
workflow.add_edge("risk_assessment_agent", "investment_advisor_agent")
workflow.add_edge("investment_advisor_agent", "budgeting_advisor_agent")
workflow.add_edge("budgeting_advisor_agent", "display_agent_response")
workflow.add_edge("display_agent_response", END)

# Entrypoint
workflow.set_entry_point("user_input")

# Compile the graph
app = workflow.compile()


# --- Streamlit UI ---
#st.title("Personalized Financial Advisor Agent Team (MultiAgent + RAG) 🏦")

#if "chat_state" not in st.session_state:
#    st.session_state.chat_state = AgentState(user_info={}, messages=[], agent_response="") # Initialize full AgentState
#    st.session_state.chat_history = []

# Initialize chat_state at the beginning of each run, ensuring it's an AgentState
#st.session_state.chat_state = AgentState(user_info={}, messages=[], agent_response="") # Default AgentState
#
#if "chat_state" not in st.session_state: # Now only initialize specific fields if not already in session_state
#    st.session_state.chat_history = []
#    print("Chat state initialized (first run)") # Keep the print for first run indication
#else:
#    print("Chat state already in session state (subsequent run)") # Indicate subsequent runs
#
#user_info_inputs = {} # Dictionary to hold user input from expander
#
#with st.expander("Tell us about yourself to personalize your financial advice"):
#   user_info_inputs['risk_tolerance'] = st.selectbox("What is your risk tolerance?", ["Low", "Medium", "High", "Unknown"])
#    user_info_inputs['financial_goals'] = st.text_input("What are your financial goals? (e.g., retirement, buying a house)")
#    user_info_inputs['income'] = st.number_input("What is your monthly income?", value=0.0)
#   user_info_inputs['expenses'] = st.number_input("What are your monthly expenses?", value=0.0)
#
#user_prompt = st.chat_input("Ask for financial advice:")
#
#if user_prompt:
#    st.session_state.chat_state.user_info = user_info_inputs # Update user_info in state
#    st.session_state.chat_state.user_info["current_message"] = user_prompt # Add current user message to user_info
#
#    # Run LangGraph application
#    inputs = {"user_info": st.session_state.chat_state.user_info, "messages": st.session_state.chat_state.messages, "agent_response": st.session_state.chat_state.agent_response}
#    with st.spinner("Advising... (MultiAgent with RAG)"):
#        updated_state = app.invoke(inputs, config=RunnableConfig(verbose=False)) # Invoke graph, verbose=False for cleaner UI
#        print(f"Type of updated_state after app.invoke(): {type(updated_state)}") # <-- ADD THIS PRINT STATEMENT
#       # st.session_state.chat_state = updated_state # Update chat state
#        st.session_state.chat_state['user_info'] = updated_state['user_info'] # Line 319 was here, now moved below print
#        st.session_state.chat_state['messages'] = updated_state['messages']
#        st.session_state.chat_state['agent_response'] = updated_state['agent_response']
#
#    # Display chat history and agent response
#    st.session_state.chat_history.append(("User", user_prompt))
#    agent_message_content = st.session_state.chat_state.agent_response # Get agent response from state
#    st.session_state.chat_history.append(("Agent", agent_message_content))
#
#
#    for sender, message in st.session_state.chat_history:
#        if sender == "User":
#            with st.chat_message("user"):
#                st.write(message)
#        elif sender == "Agent":
#            with st.chat_message("assistant"):
#                st.write(message)
#
#    # Clear user input - optional for chat_input
#    st.session_state.chat_state.user_info["current_message"] = None
#

##########################################################################
## New Streamlit UI ##
# --- Streamlit UI ---
st.title("Personalized Financial Advisor Agent Team (MultiAgent + RAG) 🏦")

if "chat_state" not in st.session_state:
    st.session_state.chat_state = AgentState(user_info={}, messages=[], agent_response="")
    st.session_state.chat_history = []

user_info_inputs = {} # Dictionary to hold user input from expander

with st.expander("Please Tell us about yourself to personalize your financial advice"):
    user_info_inputs['risk_tolerance'] = st.selectbox("What is your risk tolerance?", ["Low", "Medium", "High", "Unknown"])
    user_info_inputs['financial_goals'] = st.text_input("What are your financial goals? (e.g., retirement, buying a house)")
    user_info_inputs['income'] = st.number_input("What is your monthly income?", value=0.0)
    user_info_inputs['expenses'] = st.number_input("What are your monthly expenses?", value=0.0)

user_prompt = st.chat_input("Ask for financial advice:")

if user_prompt:
    # Correctly update user_info within the AgentState object
    st.session_state.chat_state['user_info'] = user_info_inputs
    st.session_state.chat_state['user_info']['current_message'] = user_prompt # Add current message

    # Run LangGraph application - Pass the full AgentState
    inputs = st.session_state.chat_state
    with st.spinner("Advising... (MultiAgent with RAG)"):
        updated_state = app.invoke(inputs, config=RunnableConfig(verbose=False))
        # Correctly update the entire chat_state with the output of app.invoke()
        st.session_state.chat_state = updated_state

    # Display chat history and agent response
    st.session_state.chat_history.append(("User", user_prompt))
    agent_message_content = st.session_state.chat_state['agent_response'] # Access agent_response from AgentState
    st.session_state.chat_history.append(("Agent", agent_message_content))

    for sender, message in st.session_state.chat_history:
        if sender == "User":
            with st.chat_message("user"):
                st.write(message)
        elif sender == "Agent":
            with st.chat_message("assistant"):
                st.write(message)

    # Clear user input - optional for chat_input - No need to clear current_message anymore, handled in next user input