"""LangGraph workflow for Jeep Patriot diagnostic assistant."""

from typing import Dict, Any, List, TypedDict
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from pdf_reader import PatriotManualReader
import logging

logger = logging.getLogger(__name__)

class PatriotDiagnosticState(TypedDict):
    """State management for the diagnostic workflow."""
    user_query: str
    manual_content: Dict[str, str]
    relevant_sections: List[str]
    diagnosis: str
    recommendations: List[str]
    conversation_history: List[Dict]

class PatriotAgent:
    """Main agent for Jeep Patriot diagnostics using LangGraph."""
    
    def __init__(self, pdf_path: str):
        self.pdf_reader = PatriotManualReader(pdf_path)
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1
        )
        self.workflow = self._build_workflow()
        
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(PatriotDiagnosticState)
        
        # Add nodes
        workflow.add_node("read_manual", self._read_manual_node)
        workflow.add_node("analyze_query", self._analyze_query_node)
        workflow.add_node("search_manual", self._search_manual_node)
        workflow.add_node("generate_diagnosis", self._generate_diagnosis_node)
        workflow.add_node("format_response", self._format_response_node)
        
        # Define the flow
        workflow.set_entry_point("read_manual")
        workflow.add_edge("read_manual", "analyze_query")
        workflow.add_edge("analyze_query", "search_manual")
        workflow.add_edge("search_manual", "generate_diagnosis")
        workflow.add_edge("generate_diagnosis", "format_response")
        workflow.add_edge("format_response", END)
        
        return workflow.compile()
    
    def _read_manual_node(self, state: PatriotDiagnosticState) -> Dict[str, Any]:
        """Read and process the Jeep Patriot manual."""
        logger.info("Reading Jeep Patriot manual...")
        
        try:
            state["manual_content"] = self.pdf_reader.extract_sections()
            logger.info(f"Successfully loaded manual with {len(state['manual_content'])} sections")
        except Exception as e:
            logger.error(f"Error reading manual: {e}")
            state["manual_content"] = {}
        
        return state
    
    def _analyze_query_node(self, state: PatriotDiagnosticState) -> Dict[str, Any]:
        """Analyze user query to understand the issue."""
        logger.info("Analyzing user query...")
        
        analysis_prompt = f"""
        Analyze this Jeep Patriot related query: "{state['user_query']}"
        
        Identify:
        1. The main system involved (engine, transmission, electrical, brakes, etc.)
        2. Symptoms described
        3. Potential diagnostic areas to focus on
        
        Respond with a brief analysis in JSON format:
        {{
            "system": "primary system name",
            "symptoms": ["symptom1", "symptom2"],
            "focus_areas": ["area1", "area2"]
        }}
        """
        
        try:
            response = self.llm.invoke([
                SystemMessage(content="You are an expert automotive diagnostic assistant specializing in Jeep Patriots."),
                HumanMessage(content=analysis_prompt)
            ])
            
            # Store the analysis (simplified for now)
            if "conversation_history" not in state:
                state["conversation_history"] = []
            state["conversation_history"].append({
                "type": "analysis",
                "content": response.content
            })
            
        except Exception as e:
            logger.error(f"Error analyzing query: {e}")
        
        return state
    
    def _search_manual_node(self, state: PatriotDiagnosticState) -> Dict[str, Any]:
        """Search manual for relevant information."""
        logger.info("Searching manual for relevant information...")
        
        # Search for query-related content
        search_results = self.pdf_reader.search_manual(state["user_query"])
        
        # Also search for common automotive terms
        automotive_terms = ["diagnostic", "troubleshoot", "symptom", "repair", "maintenance"]
        for term in automotive_terms:
            if term.lower() in state["user_query"].lower():
                additional_results = self.pdf_reader.search_manual(term)
                search_results.extend(additional_results[:3])  # Limit additional results
        
        state["relevant_sections"] = search_results[:15]  # Limit total results
        logger.info(f"Found {len(state['relevant_sections'])} relevant manual sections")
        
        return state
    
    def _generate_diagnosis_node(self, state: PatriotDiagnosticState) -> Dict[str, Any]:
        """Generate diagnosis based on manual content and query."""
        logger.info("Generating diagnosis...")
        
        # Check if we have any relevant manual sections
        if not state["relevant_sections"] or len(state["relevant_sections"]) == 0:
            state["diagnosis"] = f"""I apologize, but I cannot find any information about "{state['user_query']}" in the 2011 Jeep Patriot manual. 

I can only provide diagnostic assistance based on the official manual content. Please try rephrasing your question using specific automotive terms like:
- Engine problems
- Transmission issues  
- Brake concerns
- Electrical problems
- Cooling system
- Maintenance procedures

Or describe specific symptoms you're experiencing with your Patriot."""
            return state
        
        manual_context = "\n\n".join(state["relevant_sections"])
        
        diagnosis_prompt = f"""
        Based ONLY on the Jeep Patriot manual content below, provide a diagnostic response for this query:
        
        USER QUERY: {state['user_query']}
        
        RELEVANT MANUAL CONTENT:
        {manual_context}
        
        IMPORTANT: Only use information from the manual content provided above. Do not add general automotive knowledge.
        
        Provide a response that includes:
        1. What the manual says about this issue
        2. Manual-specified diagnostic steps
        3. Manual-recommended solutions
        4. Any safety warnings from the manual
        
        If the manual content doesn't fully address the query, state that clearly.
        """
        
        try:
            response = self.llm.invoke([
                SystemMessage(content="""You are a Jeep Patriot manual assistant. 
                ONLY use the provided manual content in your response. 
                Do not add general automotive knowledge or advice not found in the manual.
                If the manual doesn't contain enough information, say so clearly."""),
                HumanMessage(content=diagnosis_prompt)
            ])
            
            state["diagnosis"] = response.content
            
        except Exception as e:
            logger.error(f"Error generating diagnosis: {e}")
            state["diagnosis"] = "I apologize, but I encountered an error while analyzing the manual content. Please try again."
        
        return state
    
    def _format_response_node(self, state: PatriotDiagnosticState) -> Dict[str, Any]:
        """Format the final response."""
        logger.info("Formatting response...")
        
        # The diagnosis is already well-formatted from the LLM
        # Add any additional formatting if needed
        
        return state
    
    def diagnose(self, user_query: str) -> str:
        """Main method to diagnose Jeep Patriot issues."""
        state = {
            "user_query": user_query,
            "manual_content": {},
            "relevant_sections": [],
            "diagnosis": "",
            "recommendations": [],
            "conversation_history": []
        }
        
        # Run the workflow
        final_state = self.workflow.invoke(state)
        
        return final_state["diagnosis"]
