import os
import time
import pandas as pd
from dotenv import load_dotenv
from typing import TypedDict, Literal
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# ==========================================
# 1. DEFINE THE SYSTEM MEMORY (STATE)
# ==========================================
class TicketState(TypedDict):
    ticket_id: int
    customer_message: str
    category: str
    resolution_draft: str
    requires_human: bool

# Pydantic model updated for Banking/Fintech Support
class IntentClassification(BaseModel):
    category: Literal["Card Issue", "Transaction Query", "Account Access", "Other"] = Field(
        description="The main category of the customer's request."
    )
    is_urgent: bool = Field(description="True if the customer is panicked, angry, or reporting fraud/stolen items.")

# ==========================================
# 2. INITIALIZE THE LLM
# ==========================================
# Using the "latest" tag ensures the API won't break when Google updates models
llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0)
structured_llm = llm.with_structured_output(IntentClassification)

# ==========================================
# 3. DEFINE THE AGENTS (NODES)
# ==========================================
def triage_agent(state: TicketState):
    """Agent 1: Reads the ticket and categorizes it."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert banking customer support triage agent. Classify the user's issue."),
        ("human", "{message}")
    ])
    
    chain = prompt | structured_llm
    result = chain.invoke({"message": state["customer_message"]})
    
    state["category"] = result.category
    
    # Escalate fraud, urgency, or account access issues immediately
    if result.is_urgent or result.category == "Account Access":
        state["requires_human"] = True
    else:
        state["requires_human"] = False
        
    return state

def auto_responder_agent(state: TicketState):
    """Agent 2: Drafts responses for simple, routine tickets."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a polite banking support AI. Draft a concise, professional 2-sentence response addressing the customer's {category} issue. Assure them we are looking into it."),
        ("human", "Customer Message: {message}")
    ])
    
    # StrOutputParser() removes any invisible API metadata/signatures
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"category": state["category"], "message": state["customer_message"]})
    
    state["resolution_draft"] = result
    return state

def escalation_agent(state: TicketState):
    """Agent 3: Drafts a summary for a human agent to take over."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an internal assistant. This ticket requires human intervention. Draft a 1-sentence summary of the problem for the human agent."),
        ("human", "Customer Message: {message}")
    ])
    
    # StrOutputParser() removes any invisible API metadata/signatures
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"message": state["customer_message"]})
    
    state["resolution_draft"] = f"[ESCALATED TO HUMAN RISK TEAM] Summary: {result}"
    return state

# ==========================================
# 4. DEFINE THE ROUTING LOGIC
# ==========================================
def route_ticket(state: TicketState) -> str:
    """Decides which agent gets the ticket next."""
    if state["requires_human"]:
        return "escalate"
    else:
        return "auto_respond"

# ==========================================
# 5. BUILD THE GRAPH (THE WORKFLOW)
# ==========================================
workflow = StateGraph(TicketState)

workflow.add_node("triage", triage_agent)
workflow.add_node("auto_respond", auto_responder_agent)
workflow.add_node("escalate", escalation_agent)

workflow.set_entry_point("triage")
workflow.add_conditional_edges(
    "triage",
    route_ticket,
    {
        "auto_respond": "auto_respond",
        "escalate": "escalate"
    }
)
workflow.add_edge("auto_respond", END)
workflow.add_edge("escalate", END)

app = workflow.compile()

# ==========================================
# 6. RUN ON REAL DATA (BANKING77 DATASET)
# ==========================================
def main():
    print("Downloading the PolyAI Banking77 Customer Support dataset...\n")
    
    # Highly stable raw GitHub URL for the open-source dataset
    csv_url = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv"
    
    try:
        # Load the CSV and grab 5 random real-world customer queries
        df = pd.read_csv(csv_url).sample(n=5, random_state=42)
        sample_tickets = df.to_dict('records')
    except Exception as e:
        print(f"Error downloading data. Check your internet connection. Details: {e}")
        return
    
    print("-" * 60)
    
    for idx, row in enumerate(sample_tickets):
        # Initialize the state using the 'text' column from the Banking77 dataset
        initial_state = {
            "ticket_id": idx + 1,
            "customer_message": row["text"], 
            "category": "",
            "resolution_draft": "",
            "requires_human": False
        }
        
        # Run the agentic workflow
        final_state = app.invoke(initial_state)
        
        # Print the results
        print(f"🎟️  TICKET #{final_state['ticket_id']}")
        print(f"🗣️  CUSTOMER: '{final_state['customer_message']}'")
        print(f"📂  AI CATEGORY: {final_state['category']}")
        print(f"🚨  REQUIRES HUMAN: {final_state['requires_human']}")
        print(f"🤖  AGENT ACTION:\n{final_state['resolution_draft']}")
        print("-" * 60)
        
        # Pause for 15 seconds to respect the free API rate limit
        if idx < len(sample_tickets) - 1:
            print("Pausing for 15 seconds to respect API rate limits...")
            time.sleep(15)
            print("\nProcessing next ticket...\n")

if __name__ == "__main__":
    main()