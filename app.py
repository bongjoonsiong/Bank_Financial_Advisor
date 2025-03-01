# Preload pysqlite3-binary to override system SQLite before any Chroma imports
import sys
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    import streamlit as st
    st.error("pysqlite3-binary not installed. Please add it to requirements.txt.")
    st.stop()

import streamlit as st
import os
import chromadb
import json
import re

try:
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma
    from langchain_deepseek.chat_models import ChatDeepSeek
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.messages import AIMessage, HumanMessage
    from langchain.tools import tool
    from langchain_community.document_loaders import PyPDFLoader
    from langgraph.graph import StateGraph, END
    from typing import TypedDict, Dict, Any, List
except ImportError as e:
    st.error(f"Import error: {e}. Please check your dependencies.")
    st.stop()

# --- Page Configuration ---
st.set_page_config(
    page_title="Personalized Financial Advisor (MultiAgent + RAG)",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Sidebar ---
with st.sidebar:
    try:
        st.image("Bank_Financial_Advisor_Favicon.ico")
    except FileNotFoundError:
        st.warning("Favicon not found. Skipping image load.")
    st.title("Personalized Financial Advisor (MultiAgent + RAG)")
    st.markdown("MultiAgent Chatbot with RAG for personalized financial advice.")
    st.markdown("---")
    st.markdown("## Agent-Based Architecture")
    st.markdown("- **Personalization Agent:** Gathers user info.")
    st.markdown("- **Risk Assessment Agent:** Assesses risk tolerance.")
    st.markdown("- **Investment Advisor Agent:** Provides advice with RAG.")
    st.markdown("- **Budgeting Advisor Agent:** Suggests budgeting tips.")

# --- Global Variables and Functions ---

embedding_model_name = "all-MiniLM-L6-v2"
try:
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
except Exception as e:
    st.error(f"Error initializing embeddings: {e}")
    st.stop()
persist_directory = "chroma_db"

def load_vector_db():
    try:
        client = chromadb.PersistentClient(path=persist_directory)
        vectordb = Chroma(
            client=client,
            embedding_function=embeddings,
            collection_name="bank_info"
        )
        print(f"ChromaDB loaded from: {persist_directory}")
        if not vectordb.similarity_search("investment products", k=1):
            loader = PyPDFLoader("Comprehensive_Bank_Information.pdf")
            documents = loader.load()
            vectordb.add_documents(documents)
            print("Populated ChromaDB with Comprehensive_Bank_Information.pdf data.")
        return vectordb
    except Exception as e:
        st.error(f"Error loading ChromaDB or PDF: {e}")
        return None

def load_deepseek_llm():
    try:
        deepseek_api_key = st.secrets.get("DEEPSEEK_API_KEY")
        if not deepseek_api_key:
            raise ValueError("DEEPSEEK_API_KEY not found in secrets.")
        llm = ChatDeepSeek(
            api_key=deepseek_api_key,
            model="deepseek-chat",
            temperature=0.7,
            top_p=0.9,
            max_tokens=1000,
        )
        test_response = llm.invoke("Test: Are you working?")
        st.write("DeepSeek LLM Test Response:", test_response.content)
        return llm
    except Exception as e:
        st.warning(f"Error initializing DeepSeek LLM: {e}. Using dummy mode.")
        class DummyLLM:
            def invoke(self, prompt):
                return AIMessage(content="Dummy response: DeepSeek unavailable.")
        return DummyLLM()

vectordb_global = load_vector_db()
llm_global = load_deepseek_llm()

if llm_global is None or vectordb_global is None:
    st.warning("Some components failed to initialize, but proceeding with limited functionality.")

# --- Tool Definitions ---

@tool
def get_user_risk_tolerance(user_info: Dict) -> str:
    """Returns the user's risk tolerance based on provided user information."""
    return f"User's risk tolerance is: {user_info.get('risk_tolerance', 'Unknown')}"

@tool
def get_user_financial_goals(user_info: Dict) -> str:
    """Returns the user's financial goals based on provided user information."""
    return f"User's financial goals are: {user_info.get('financial_goals', 'Unknown')}"

@tool
def provide_investment_advice_rag(risk_tolerance: str, financial_goals: str, query: str) -> str:
    """Provides investment advice based on risk tolerance, financial goals, and retrieved OCBC bank product info."""
    if vectordb_global is None:
        return "Vector database not initialized."
    retrieved_info = perform_vector_search(query)
    advice_prompt = f"""
    Based on risk tolerance: '{risk_tolerance}', financial goals: '{financial_goals}',
    and OCBC bank product info retrieved from ChromaDB:\n{retrieved_info}\n
    Provide concise investment advice, recommending specific OCBC products matching the query '{query}' if relevant.
    """
    response = llm_global.invoke(advice_prompt)
    return response.content

@tool
def suggest_budgeting_tips(income: float, expenses: float) -> str:
    """Suggests budgeting tips based on the user's income and expenses."""
    prompt = f"Income: ${income}, Expenses: ${expenses}. Suggest budgeting tips."
    response = llm_global.invoke(prompt)
    return response.content

def perform_vector_search(query: str) -> str:
    if vectordb_global is None:
        return "Vector database not initialized."
    try:
        results = vectordb_global.similarity_search(query, k=3)
        return "\n\n".join([doc.page_content for doc in results]) or "No relevant OCBC product info found."
    except Exception as e:
        return f"Vector search error: {e}"

# Tools
personalization_tools = {
    "get_user_risk_tolerance": get_user_risk_tolerance,
    "get_user_financial_goals": get_user_financial_goals,
    "provide_investment_advice_rag": provide_investment_advice_rag
}
risk_assessment_tools = {"get_user_risk_tolerance": get_user_risk_tolerance}
investment_tools = {"provide_investment_advice_rag": provide_investment_advice_rag, "get_user_financial_goals": get_user_financial_goals, "get_user_risk_tolerance": get_user_risk_tolerance}
budgeting_tools = {"suggest_budgeting_tips": suggest_budgeting_tips}

# --- AgentState ---
class AgentState(TypedDict):
    user_info: Dict[str, Any]
    messages: List[AIMessage | HumanMessage]
    agent_response: str

# --- Agent Logic with Enhanced Parsing ---
def run_agent(messages: List, system_prompt: str, tools: Dict[str, Any], user_info: Dict) -> str:
    user_info_str = f"User Info: Risk Tolerance: {user_info.get('risk_tolerance', 'Unknown')}, Financial Goals: {user_info.get('financial_goals', 'Unknown')}, Income: {user_info.get('income', 0.0)}, Expenses: {user_info.get('expenses', 0.0)}"
    latest_query = messages[-1].content if messages else "No query provided"
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"{system_prompt}\n{user_info_str}\nAvailable tools: {list(tools.keys())}.\nFor investment queries, use '[provide_investment_advice_rag]' with the query '{latest_query}'. For budgeting, use '[suggest_budgeting_tips]'. Otherwise, use '[get_user_risk_tolerance]' or '[get_user_financial_goals]' as needed. Use '[tool_name]' syntax only—no JSON or plain text 'Action:' format."),
        *messages
    ])
    try:
        response = llm_global.invoke(prompt.format())
        response_text = response.content.strip()  # Strip whitespace to avoid parsing issues
        print(f"DEBUG: LLM Response: {response_text}")
    except Exception as e:
        st.error(f"Error invoking LLM: {e}")
        return f"Error invoking LLM: {e}"

    # Parse [tool_name], JSON, and plain text "Action: tool_name"
    tool_calls = re.findall(r'\[(.*?)\]', response_text)
    if not tool_calls:
        json_match = re.search(r'\{.*"tool":\s*"([^"]+)".*\}', response_text, re.DOTALL)
        if json_match:
            tool_calls = [json_match.group(1)]
        else:
            action_match = re.search(r'Action:\s*(\w+)', response_text)
            if action_match:
                tool_calls = [action_match.group(1)]

    if tool_calls:
        for tool_name in tool_calls:
            if tool_name in tools:
                try:
                    if tool_name == "provide_investment_advice_rag":
                        result = tools[tool_name].invoke({
                            "risk_tolerance": user_info.get("risk_tolerance", "Unknown"),
                            "financial_goals": user_info.get("financial_goals", "Unknown"),
                            "query": latest_query
                        })
                    elif tool_name == "suggest_budgeting_tips":
                        result = tools[tool_name].invoke({
                            "income": user_info.get("income", 0.0),
                            "expenses": user_info.get("expenses", 0.0)
                        })
                    else:
                        result = tools[tool_name].invoke({"user_info": user_info})
                    print(f"DEBUG: Tool {tool_name} Result: {result}")
                    return result  # Return only the tool result for clarity
                except Exception as e:
                    st.error(f"Error invoking tool {tool_name}: {e}")
                    return f"Error invoking tool {tool_name}: {e}"
            else:
                st.error(f"Invalid tool name in LLM response: {tool_name}")
                return f"Invalid tool name: {tool_name}"
    return response_text

# --- Node Functions ---
def user_input_node(state: AgentState):
    user_message = state['user_info'].get("current_message")
    return {"messages": state['messages'] + ([HumanMessage(content=user_message)] if user_message else [])}

def personalization_node(state: AgentState):
    response = run_agent(
        state['messages'],
        "You are a Personalization Agent. Gather or confirm user financial info (risk tolerance, goals, income, expenses). If the query involves investments, directly use '[provide_investment_advice_rag]'.",
        personalization_tools,
        state['user_info']
    )
    return {"messages": state['messages'] + [AIMessage(content=response)]}

def risk_assessment_node(state: AgentState):
    response = run_agent(
        state['messages'],
        "You are a Risk Assessment Agent. Assess or confirm the user's risk tolerance.",
        risk_assessment_tools,
        state['user_info']
    )
    return {"messages": state['messages'] + [AIMessage(content=response)]}

def investment_node(state: AgentState):
    latest_query = state['messages'][-1].content if state['messages'] else "No query provided"
    if "investment" in latest_query.lower() or "products" in latest_query.lower():
        result = investment_tools["provide_investment_advice_rag"].invoke({
            "risk_tolerance": state['user_info'].get("risk_tolerance", "Unknown"),
            "financial_goals": state['user_info'].get("financial_goals", "Unknown"),
            "query": latest_query
        })
        response = result  # Use tool result directly
    else:
        response = run_agent(
            state['messages'],
            "You are an Investment Advisor Agent. Answer investment queries using risk tolerance, goals, and OCBC product info from RAG. Use '[provide_investment_advice_rag]' for product recommendations.",
            investment_tools,
            state['user_info']
        )
    return {"messages": state['messages'] + [AIMessage(content=response)]}

def budgeting_node(state: AgentState):
    response = run_agent(
        state['messages'],
        "You are a Budgeting Advisor Agent. Suggest budgeting tips based on income and expenses if relevant. Use '[suggest_budgeting_tips]' if appropriate.",
        budgeting_tools,
        state['user_info']
    )
    return {"messages": state['messages'] + [AIMessage(content=response)]}

def agent_response_node(state: AgentState):
    # Aggregate all tool results into a single response
    full_response = []
    seen_tools = set()  # Avoid duplicates
    for msg in state['messages']:
        if isinstance(msg, AIMessage):
            if "Tool Result:" in msg.content:
                tool_result = msg.content.split("Tool Result:")[-1].strip()
                full_response.append(tool_result)
            elif any(tool in msg.content for tool in ["provide_investment_advice_rag", "suggest_budgeting_tips"]):
                # Extract tool calls from text if not already processed
                tool_calls = re.findall(r'\[(.*?)\]', msg.content)
                for tool_name in tool_calls:
                    if tool_name not in seen_tools and tool_name in investment_tools:
                        seen_tools.add(tool_name)
                        if tool_name == "provide_investment_advice_rag":
                            result = investment_tools[tool_name].invoke({
                                "risk_tolerance": state['user_info'].get("risk_tolerance", "Unknown"),
                                "financial_goals": state['user_info'].get("financial_goals", "Unknown"),
                                "query": state['user_info'].get("current_message", "No query provided")
                            })
                            full_response.append(result)
                        elif tool_name == "suggest_budgeting_tips":
                            result = budgeting_tools[tool_name].invoke({
                                "income": state['user_info'].get("income", 0.0),
                                "expenses": state['user_info'].get("expenses", 0.0)
                            })
                            full_response.append(result)
            else:
                full_response.append(msg.content)
    return {"agent_response": "\n\n".join(full_response).strip() if full_response else ""}

# --- Workflow ---
workflow = StateGraph(AgentState)
workflow.add_node("user_input", user_input_node)
workflow.add_node("personalization", personalization_node)
workflow.add_node("risk_assessment", risk_assessment_node)
workflow.add_node("investment", investment_node)
workflow.add_node("budgeting", budgeting_node)
workflow.add_node("display_response", agent_response_node)

workflow.add_edge("user_input", "personalization")
workflow.add_edge("personalization", "risk_assessment")
workflow.add_edge("risk_assessment", "investment")
workflow.add_edge("investment", "budgeting")
workflow.add_edge("budgeting", "display_response")
workflow.add_edge("display_response", END)

workflow.set_entry_point("user_input")
app = workflow.compile()

# --- Streamlit UI ---
st.title("Personalized Financial Advisor (MultiAgent + RAG) 🏦")

if "chat_state" not in st.session_state:
    st.session_state.chat_state = AgentState(user_info={}, messages=[], agent_response="")
    st.session_state.chat_history = []

with st.expander("Tell us about yourself"):
    user_info = {
        'risk_tolerance': st.selectbox("Risk tolerance?", ["Low", "Medium", "High", "Unknown"]),
        'financial_goals': st.text_input("Financial goals?"),
        'income': st.number_input("Monthly income?", value=0.0),
        'expenses': st.number_input("Monthly expenses?", value=0.0)
    }

user_prompt = st.chat_input("Ask for financial advice:")

if user_prompt:
    st.session_state.chat_state['user_info'] = user_info
    st.session_state.chat_state['user_info']['current_message'] = user_prompt
    
    with st.spinner("Advising..."):
        try:
            updated_state = app.invoke(st.session_state.chat_state)
            st.session_state.chat_state = updated_state
            st.success("Advising complete!")
        except Exception as e:
            st.error(f"Workflow error: {str(e)}")
            st.write(f"DEBUG: Full error traceback: {repr(e)}")
            st.stop()
    
    st.session_state.chat_history.append(("User", user_prompt))
    agent_response = st.session_state.chat_state['agent_response']
    st.session_state.chat_history.append(("Agent", agent_response))

    for sender, message in st.session_state.chat_history:
        with st.chat_message("user" if sender == "User" else "assistant"):
            st.write(message)