from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, TypedDict
from enum import Enum
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.document_loaders import WebBaseLoader
from bs4 import BeautifulSoup
import requests
import os
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI(title="Coding Ninja Multi-Agent System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic Models
class UserQuery(BaseModel):
    text: str
    user_id: Optional[str] = None
    context: Optional[Dict] = None

class AgentType(str, Enum):
    CUSTOMER_SUPPORT = "customer_support"
    CONTENT_FETCHER = "content_fetcher"
    CODE_HELPER = "code_helper"
    COURSE_ADVISOR = "course_advisor"
    INTERVIEW_PREP = "interview_prep"
    PROJECT_IDEAS = "project_ideas"

class AgentResponse(BaseModel):
    content: str
    agent_type: AgentType
    sources: Optional[List[str]] = None
    suggested_actions: Optional[List[str]] = None

class MultiAgentResponse(BaseModel):
    responses: List[AgentResponse]
    summary: str

# State Definition
class AgentState(TypedDict):
    messages: List[BaseMessage]
    next_node: Optional[str] = None

# Initialize Groq LLM
groq_llm = ChatGroq(
    temperature=0.7,
    model_name="deepseek-r1-distill-llama-70b",
    api_key=os.getenv("key")
)

# Initialize Tools
search_tool = DuckDuckGoSearchRun()

url = "https://www.codingninjas.com/"
def fetch_coding_ninja_content(url: str):
    """Custom fetcher for Coding Ninja website"""
    try:
        loader = WebBaseLoader(url)
        docs = loader.load()
        soup = BeautifulSoup(docs[0].page_content, 'html.parser')
        for element in soup(['script', 'style', 'nav', 'footer']):
            element.decompose()
        return " ".join(soup.stripped_strings)
    except Exception as e:
        return f"Error fetching content: {str(e)}"

# Agent Prompts (examples)
CUSTOMER_SUPPORT_PROMPT = ChatPromptTemplate.from_template(
    """You are a Coding Ninja customer support specialist. Answer:
    {question}"""
)

# Agent Nodes
async def customer_support_node(state: AgentState):
    question = state["messages"][-1].content
    chain = CUSTOMER_SUPPORT_PROMPT | groq_llm
    response = await chain.ainvoke({"question": question})
    return {
        "messages": state["messages"] + [
            AIMessage(
                content=response.content,
                additional_kwargs={"agent_type": AgentType.CUSTOMER_SUPPORT}
            )
        ],
        "next_node": None
    }

async def content_fetcher_node(state: AgentState):
    question = state["messages"][-1].content
    search_results = search_tool.run(f"site:https://www.codingninjas.com/ {question}")
    content = fetch_coding_ninja_content(search_results.split()[0]) if search_results else "No content found"
    
    chain = ChatPromptTemplate.from_template("Summarize: {content}") | groq_llm
    response = await chain.ainvoke({"content": content})
    return {
        "messages": state["messages"] + [
            AIMessage(
                content=response.content,
                additional_kwargs={
                    "agent_type": AgentType.CONTENT_FETCHER,
                    "sources": [search_results.split()[0]] if search_results else []
                }
            )
        ],
        "next_node": None
    }

# Agent Nodes Implementation
async def code_helper_node(state: AgentState):
    question = state["messages"][-1].content
    chain = ChatPromptTemplate.from_template(
        """You are a coding expert. Help with:
        {question}
        
        Provide clear explanations and code examples:"""
    ) | groq_llm
    response = await chain.ainvoke({"question": question})
    return {
        "messages": state["messages"] + [
            AIMessage(
                content=response.content,
                additional_kwargs={"agent_type": AgentType.CODE_HELPER}
            )
        ],
        "next_node": None
    }

async def course_advisor_node(state: AgentState):
    question = state["messages"][-1].content
    chain = ChatPromptTemplate.from_template(
        """ You are course adviser of coding ninja plateform. Recommend courses based on:
        Question: {question}
        Skill Level: intermediate
        
        Provide course names, durations and learning outcomes:"""
    ) | groq_llm
    response = await chain.ainvoke({"question": question})
    return {
        "messages": state["messages"] + [
            AIMessage(
                content=response.content,
                additional_kwargs={"agent_type": AgentType.COURSE_ADVISOR}
            )
        ],
        "next_node": None
    }

async def interview_prep_node(state: AgentState):
    question = state["messages"][-1].content
    chain = ChatPromptTemplate.from_template(
        """Prepare for technical interviews:
        Question: {question}
        Tech Stack: Python, JavaScript
        Skill Level: intermediate
        
        Provide common questions, answers, and tips:"""
    ) | groq_llm
    response = await chain.ainvoke({"question": question})
    return {
        "messages": state["messages"] + [
            AIMessage(
                content=response.content,
                additional_kwargs={"agent_type": AgentType.INTERVIEW_PREP}
            )
        ],
        "next_node": None
    }

async def project_ideas_node(state: AgentState):
    question = state["messages"][-1].content
    chain = ChatPromptTemplate.from_template(
        """Generate project ideas for:
        Technologies: Python, JavaScript, Django, React
        Skill Level: intermediate
        Interests: {question}
        
        Provide 3-5 project ideas with descriptions:"""
    ) | groq_llm
    response = await chain.ainvoke({"question": question})
    return {
        "messages": state["messages"] + [
            AIMessage(
                content=response.content,
                additional_kwargs={"agent_type": AgentType.PROJECT_IDEAS}
            )
        ],
        "next_node": None
    }

# Router Node
async def router_node(state: AgentState):
    question = state["messages"][-1].content.lower()
    if any(word in question for word in ["customer", "support", "help"]):
        return {"next_node": "customer_support"}
    elif any(word in question for word in ["course", "learn"]):
        return {"next_node": "course_advisor"}
    elif any(word in question for word in ["code", "program"]):
        return {"next_node": "code_helper"}
    elif any(word in question for word in ["interview", "prepare"]):
        return {"next_node": "interview_prep"}
    elif any(word in question for word in ["project", "idea"]):
        return {"next_node": "project_ideas"}
    else:
        return {"next_node": "content_fetcher"}

# Build the Graph
workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("router", router_node)
workflow.add_node("customer_support", customer_support_node)
workflow.add_node("content_fetcher", content_fetcher_node)
workflow.add_node("code_helper", code_helper_node)
workflow.add_node("course_advisor", course_advisor_node)
workflow.add_node("interview_prep", interview_prep_node)
workflow.add_node("project_ideas", project_ideas_node)

# Add Edges
workflow.add_edge("customer_support", END)
workflow.add_edge("content_fetcher", END)
workflow.add_edge("code_helper", END)
workflow.add_edge("course_advisor", END)
workflow.add_edge("interview_prep", END)
workflow.add_edge("project_ideas", END)

# Configure Routing
workflow.add_conditional_edges(
    "router",
    lambda state: state.get("next_node", "content_fetcher"),
    {
        "customer_support": "customer_support",
        "content_fetcher": "content_fetcher",
        "code_helper": "code_helper",
        "course_advisor": "course_advisor",
        "interview_prep": "interview_prep",
        "project_ideas": "project_ideas"
    }
)

workflow.set_entry_point("router")
agent = workflow.compile()

# API Endpoints
@app.get("/")
def read_root():
    return {"message": "Coding Ninja Multi-Agent System"}

@app.post("/query", response_model=MultiAgentResponse)
async def handle_user_query(query: UserQuery):
    try:
        # Initialize state properly
        response = await agent.ainvoke({
            "messages": [HumanMessage(content=query.text)],
            "next_node": "router"
        })
        
        last_message = response["messages"][-1]
        
        return MultiAgentResponse(
            responses=[AgentResponse(
                content=last_message.content,
                agent_type=last_message.additional_kwargs["agent_type"],
                sources=last_message.additional_kwargs.get("sources", []),
                suggested_actions=get_suggested_actions(last_message.additional_kwargs["agent_type"])
            )],
            summary=last_message.content[:200] + "..." if len(last_message.content) > 200 else last_message.content
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def get_suggested_actions(agent_type: AgentType) -> List[str]:
    actions = {
        AgentType.CUSTOMER_SUPPORT: ["Contact support", "Visit help center"],
        AgentType.CONTENT_FETCHER: ["Browse more content", "Save this"],
        AgentType.CODE_HELPER: ["Try in editor", "View docs"],
        AgentType.COURSE_ADVISOR: ["View syllabus", "Enroll now"],
        AgentType.INTERVIEW_PREP: ["Practice more", "Schedule mock"],
        AgentType.PROJECT_IDEAS: ["Save ideas", "Start project"]
    }
    return actions.get(agent_type, [])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)