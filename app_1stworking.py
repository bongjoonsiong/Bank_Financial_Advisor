import streamlit as st
import os
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma
    from langchain_deepseek.chat_models import ChatDeepSeek
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.messages import AIMessage, HumanMessage
    from langchain.tools import tool
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
        vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        print(f"ChromaDB loaded from: {persist_directory}")  # Changed to print instead of st.info
        return vectordb
    except Exception as e:
        st.error(f"Error loading ChromaDB: {e}")
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
    Provide concise investment advice, recommending specific OCBC products if relevant.
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
personalization_tools = {"get_user_risk_tolerance": get_user_risk_tolerance, "get_user_financial_goals": get_user_financial_goals}
risk_assessment_tools = {"get_user_risk_tolerance": get_user_risk_tolerance}
investment_tools = {"provide_investment_advice_rag": provide_investment_advice_rag, "get_user_financial_goals": get_user_financial_goals, "get_user_risk_tolerance": get_user_risk_tolerance}
budgeting_tools = {"suggest_budgeting_tips": suggest_budgeting_tips}

# --- AgentState ---
class AgentState(TypedDict):
    user_info: Dict[str, Any]
    messages: List[AIMessage | HumanMessage]
    agent_response: str

# --- Agent Logic with Enhanced Prompting ---
def run_agent(messages: List, system_prompt: str, tools: Dict[str, Any], user_info: Dict) -> str:
    user_info_str = f"User Info: Risk Tolerance: {user_info.get('risk_tolerance', 'Unknown')}, Financial Goals: {user_info.get('financial_goals', 'Unknown')}, Income: {user_info.get('income', 0.0)}, Expenses: {user_info.get('expenses', 0.0)}"
    latest_query = messages[-1].content if messages else "No query provided"
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"{system_prompt}\n{user_info_str}\nAvailable tools: {list(tools.keys())}.\nFor investment queries, use '[provide_investment_advice_rag]' with the query '{latest_query}'. For budgeting, use '[suggest_budgeting_tips]'. Otherwise, use '[get_user_risk_tolerance]' or '[get_user_financial_goals]' as needed. Always call a tool if applicable."),
        *messages
    ])
    response = llm_global.invoke(prompt.format())
    response_text = response.content

    import re
    tool_calls = re.findall(r'\[(.*?)\]', response_text)
    if tool_calls:
        for tool_name in tool_calls:
            if tool_name in tools:
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
                return f"{response_text}\nTool Result: {result}"
    return response_text

# --- Node Functions ---
def user_input_node(state: AgentState):
    user_message = state['user_info'].get("current_message")
    return {"messages": state['messages'] + ([HumanMessage(content=user_message)] if user_message else [])}

def personalization_node(state: AgentState):
    response = run_agent(
        state['messages'],
        "You are a Personalization Agent. Gather or confirm user financial info (risk tolerance, goals, income, expenses).",
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
    last_message = state['messages'][-1] if state['messages'] else None
    return {"agent_response": last_message.content if isinstance(last_message, AIMessage) else ""}

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
        except Exception as e:
            st.error(f"Workflow error: {e}")
            st.stop()

    st.session_state.chat_history.append(("User", user_prompt))
    agent_response = st.session_state.chat_state['agent_response']
    st.session_state.chat_history.append(("Agent", agent_response))

    for sender, message in st.session_state.chat_history:
        with st.chat_message("user" if sender == "User" else "assistant"):
            st.write(message)