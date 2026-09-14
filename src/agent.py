
from __future__ import annotations

import os
import re
from typing import Any, TypedDict

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI


# ============================================================
# IMPORTS
# ============================================================

try:
    from .rag import (
        load_chunks,
        load_faiss_index,
        load_embedding_model,
        search_policy,
        build_context,
    )

    from .tools import (
        check_employee_eligibility,
        validate_trip,
    )

    from .memory import ConversationMemory

except ImportError:
    from rag import (
        load_chunks,
        load_faiss_index,
        load_embedding_model,
        search_policy,
        build_context,
    )

    from tools import (
        check_employee_eligibility,
        validate_trip,
    )

    from memory import ConversationMemory


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

LLM_MODEL = "gemini-3.6-flash"
TOP_K = 3


# ============================================================
# LANGGRAPH STATE
# ============================================================

class AgentState(TypedDict, total=False):

    query: str

    employee_id: str | None
    route: str

    trip_type: str | None
    amount: float | None
    time: str | None

    tool_result: dict[str, Any] | None
    policy_results: list[dict[str, Any]]

    response: str
    sources: list[str]


# ============================================================
# TRAVEL POLICY AGENT
# ============================================================

class TravelPolicyAgent:

    def __init__(self):

        print("Loading Travel Policy Agent...")

        # ----------------------------------------------------
        # Load RAG components
        # ----------------------------------------------------

        self.chunks = load_chunks()

        self.index = load_faiss_index()

        self.embedding_model = load_embedding_model()

        # ----------------------------------------------------
        # Load Gemini
        # ----------------------------------------------------

        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not found. "
                "Please add it to your .env file."
            )

        self.llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL
        )

        # ----------------------------------------------------
        # Conversation memory
        # ----------------------------------------------------

        self.memory = ConversationMemory()

        # ----------------------------------------------------
        # Build LangGraph
        # ----------------------------------------------------

        self.graph = self._build_graph()

        print("Agent ready.")

    # ========================================================
    # BUILD LANGGRAPH
    # ========================================================

    def _build_graph(self):

        workflow = StateGraph(AgentState)

        # Nodes
        workflow.add_node(
            "route",
            self.route_request
        )

        workflow.add_node(
            "rag",
            self.rag_node
        )

        workflow.add_node(
            "tool",
            self.tool_node
        )

        workflow.add_node(
            "combined",
            self.combined_node
        )

        # START → ROUTE
        workflow.add_edge(
            START,
            "route"
        )

        # ROUTE → appropriate node
        workflow.add_conditional_edges(
            "route",
            self.choose_route,
            {
                "rag": "rag",
                "tool": "tool",
                "combined": "combined",
            }
        )

        # Nodes → END
        workflow.add_edge(
            "rag",
            END
        )

        workflow.add_edge(
            "tool",
            END
        )

        workflow.add_edge(
            "combined",
            END
        )

        return workflow.compile()

    # ========================================================
    # ROUTE REQUEST
    # ========================================================

    def route_request(
        self,
        state: AgentState
    ) -> AgentState:

        query = state["query"].strip()
        query_lower = query.lower()

        # ----------------------------------------------------
        # 1. Detect employee ID explicitly mentioned
        #    in CURRENT question
        # ----------------------------------------------------

        match = re.search(
            r"\bEMP\d{3}\b",
            query,
            re.IGNORECASE
        )

        explicit_employee_id = None

        if match:
            explicit_employee_id = (
                match.group(0).upper()
            )

        # ----------------------------------------------------
        # 2. Extract amount
        # ----------------------------------------------------

        amount = self.extract_amount(query)

        state["amount"] = amount

        # ----------------------------------------------------
        # 3. Extract trip type
        # ----------------------------------------------------

        trip_type = self.extract_trip_type(query)

        # ----------------------------------------------------
        # Follow-up detection
        # ----------------------------------------------------

        follow_up_keywords = [
            "what if",
            "what about",
            "how about",
            "then",
            "instead",
            "it costs",
            "it cost",
            "same trip",
            "that trip",
            "this trip",
        ]

        is_follow_up = any(
            keyword in query_lower
            for keyword in follow_up_keywords
        )

        # ----------------------------------------------------
        # Reuse previous trip type ONLY for follow-ups
        # ----------------------------------------------------

        if trip_type == "travel" and is_follow_up:

            previous_context = (
                self.memory.get_context()
            )

            previous_trip_type = (
                previous_context.get("trip_type")
            )

            if previous_trip_type:
                trip_type = previous_trip_type

        state["trip_type"] = trip_type

        # ----------------------------------------------------
        # 4. Extract time
        # ----------------------------------------------------

        time = self.extract_time(query)

        # Reuse previous time only for follow-up
        if time is None and is_follow_up:

            previous_context = (
                self.memory.get_context()
            )

            time = previous_context.get("time")

        state["time"] = time

        # ----------------------------------------------------
        # 5. Keywords
        # ----------------------------------------------------

        tool_keywords = [
            "eligible",
            "eligibility",
            "employee status",
            "employee details",
            "employee information",
            "approval status",
            "manager approval",
            "am i eligible",
            "can i take",
            "can i travel",
            "my eligibility",
            "my employee",
            "my status",
        ]

        policy_keywords = [
            "policy",
            "limit",
            "reimbursement",
            "reimburse",
            "airport",
            "travel",
            "trip",
            "expense",
            "late night",
            "night",
            "cancellation",
            "approved",
            "allowed",
            "business purpose",
            "approval",
        ]

        has_tool_keyword = any(
            keyword in query_lower
            for keyword in tool_keywords
        )

        has_policy_keyword = any(
            keyword in query_lower
            for keyword in policy_keywords
        )

        # ----------------------------------------------------
        # 6. Employee intent
        # ----------------------------------------------------

        has_explicit_employee = (
            explicit_employee_id is not None
        )

        has_employee_language = (
            has_tool_keyword
            or has_explicit_employee
        )

        has_trip_details = (
            amount is not None
            or time is not None
            or trip_type != "travel"
        )

        # ----------------------------------------------------
        # 7. Determine employee ID
        # ----------------------------------------------------

        employee_id = explicit_employee_id

        # IMPORTANT:
        #
        # Do NOT automatically use the UI employee ID
        # for a pure policy question.
        #
        # Example:
        #
        # UI employee ID = EMP001
        # Question =
        # "What is the airport reimbursement limit in India?"
        #
        # This remains a RAG-only question.
        #
        # Employee ID is used when the question actually
        # contains employee/trip intent.

        if (
            employee_id is None
            and has_employee_language
        ):

            employee_id = state.get(
                "employee_id"
            )

        # ----------------------------------------------------
        # Use memory for employee ID only when appropriate
        # ----------------------------------------------------

        if (
            employee_id is None
            and has_employee_language
            and is_follow_up
        ):

            previous_context = (
                self.memory.get_context()
            )

            employee_id = (
                previous_context.get(
                    "employee_id"
                )
            )

        state["employee_id"] = employee_id

        # ----------------------------------------------------
        # 8. ROUTING LOGIC
        # ----------------------------------------------------

        # ====================================================
        # CASE 1: Pure policy question
        # ====================================================
        #
        # Example:
        # "What is the airport reimbursement limit in India?"
        #
        # → RAG
        #

        if (
            has_policy_keyword
            and not has_explicit_employee
            and not has_tool_keyword
            and not has_trip_details
        ):

            route = "rag"

        # ====================================================
        # CASE 2: Employee eligibility/status
        # ====================================================
        #
        # Example:
        # "Is EMP001 eligible?"
        #
        # → TOOL
        #

        elif (
            employee_id is not None
            and has_tool_keyword
            and not has_trip_details
        ):

            route = "tool"

        # ====================================================
        # CASE 3: Employee + trip question
        # ====================================================
        #
        # Example:
        # "Can EMP001 take an airport trip?"
        #
        # → COMBINED
        #

        elif (
            employee_id is not None
            and has_explicit_employee
            and has_trip_details
        ):

            route = "combined"

        # ====================================================
        # CASE 4: Employee + policy question
        # ====================================================
        #
        # Example:
        # "What is EMP001's airport reimbursement limit?"
        #
        # → COMBINED
        #

        elif (
            employee_id is not None
            and has_explicit_employee
            and has_policy_keyword
        ):

            route = "combined"

        # ====================================================
        # CASE 5: Follow-up with amount
        # ====================================================
        #
        # Example:
        # Q1: Can EMP001 take an airport trip?
        # Q2: What if it costs ₹2,500?
        #
        # Memory supplies employee/trip context.
        #
        # → COMBINED
        #

        elif (
            employee_id is not None
            and amount is not None
            and is_follow_up
        ):

            route = "combined"

        # ====================================================
        # CASE 6: Employee tool request
        # ====================================================

        elif (
            employee_id is not None
            and has_tool_keyword
        ):

            route = "tool"

        # ====================================================
        # CASE 7: General policy question
        # ====================================================

        elif has_policy_keyword:

            route = "rag"

        # ====================================================
        # CASE 8: Default
        # ====================================================

        else:

            route = "rag"

        state["route"] = route

        return state

    # ========================================================
    # CHOOSE ROUTE
    # ========================================================

    def choose_route(
        self,
        state: AgentState
    ) -> str:

        return state["route"]

    # ========================================================
    # RAG NODE
    # ========================================================

    def rag_node(
        self,
        state: AgentState
    ) -> AgentState:

        query = state["query"]

        # ----------------------------------------------------
        # Search policy
        # ----------------------------------------------------

        results = search_policy(
            query=query,
            model=self.embedding_model,
            index=self.index,
            chunks=self.chunks,
            top_k=TOP_K
        )

        state["policy_results"] = results

        # ----------------------------------------------------
        # No relevant policy found
        # ----------------------------------------------------

        if not results:

            state["response"] = (
                "I couldn't find relevant information "
                "in the company travel policies."
            )

            state["sources"] = []

            self.save_memory(state)

            return state

        # ----------------------------------------------------
        # Build context
        # ----------------------------------------------------

        context = build_context(results)

        # ----------------------------------------------------
        # Gemini prompt
        # ----------------------------------------------------

        prompt = f"""
You are a corporate travel policy assistant.

Answer the question using ONLY the policy context
provided below.

IMPORTANT:
- This is a policy-focused question.
- Do not assume or mention any employee ID.
- Do not provide employee-specific eligibility information
  unless the question explicitly asks about an employee.
- Do not provide trip-specific validation unless the
  question explicitly asks for it.

Rules:
- Do not invent policy information.
- Do not invent reimbursement limits.
- Do not invent fees.
- Do not invent approval records.
- Do not invent exceptions.
- If the answer is not available in the context,
  clearly say that it is not specified.
- Preserve the correct currency.
- Keep the answer concise and practical.
- Mention the relevant policy source.

POLICY CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
"""

        # ----------------------------------------------------
        # Gemini call
        # ----------------------------------------------------

        response = self.llm.invoke(prompt)

        state["response"] = (
            self.extract_response_text(response)
        )

        # ----------------------------------------------------
        # Sources
        # ----------------------------------------------------

        state["sources"] = (
            self.unique_sources(results)
        )

        self.save_memory(state)

        return state

    # ========================================================
    # TOOL NODE
    # ========================================================

    def tool_node(
        self,
        state: AgentState
    ) -> AgentState:

        employee_id = state.get("employee_id")

        # ----------------------------------------------------
        # Missing employee ID
        # ----------------------------------------------------

        if not employee_id:

            state["response"] = (
                "Please provide your employee ID "
                "so I can check your eligibility."
            )

            state["sources"] = []

            return state

        # ----------------------------------------------------
        # Check employee
        # ----------------------------------------------------

        result = check_employee_eligibility(
            employee_id
        )

        state["tool_result"] = result

        # ----------------------------------------------------
        # Unknown employee
        # ----------------------------------------------------

        if not result.get("found"):

            state["response"] = (
                f"I couldn't find employee ID "
                f"{employee_id} in the employee "
                "eligibility records."
            )

            state["sources"] = [
                "employee_eligibility.txt"
            ]

            return state

        # ----------------------------------------------------
        # Build response
        # ----------------------------------------------------

        status = result["status"]

        country = result["country"]

        employee_type = result["employee_type"]

        manager_approval = result["manager_approval"]

        state["response"] = (
            f"Employee {employee_id}: "
            f"{status}. "
            f"Country: {country}. "
            f"Employee type: {employee_type}. "
            f"Manager approval: {manager_approval}."
        )

        state["sources"] = [
            "employee_eligibility.txt"
        ]

        self.save_memory(state)

        return state

    # ========================================================
    # COMBINED NODE
    # ========================================================

    def combined_node(
        self,
        state: AgentState
    ) -> AgentState:

        employee_id = state.get("employee_id")

        query = state["query"]

        # ----------------------------------------------------
        # Missing employee ID
        # ----------------------------------------------------

        if not employee_id:

            state["response"] = (
                "Please provide an employee ID "
                "so I can validate the travel request."
            )

            state["sources"] = []

            return state

        # ----------------------------------------------------
        # Employee eligibility
        # ----------------------------------------------------

        employee_result = (
            check_employee_eligibility(
                employee_id
            )
        )

        # ----------------------------------------------------
        # Unknown employee
        # ----------------------------------------------------

        if not employee_result.get("found"):

            state["response"] = (
                f"I couldn't find employee ID "
                f"{employee_id} in the employee "
                "eligibility records."
            )

            state["sources"] = [
                "employee_eligibility.txt"
            ]

            return state

        # ----------------------------------------------------
        # Get trip information
        # ----------------------------------------------------

        trip_type = state.get("trip_type")

        amount = state.get("amount")

        time = state.get("time")

        # ----------------------------------------------------
        # Validate trip if amount exists
        # ----------------------------------------------------

        validation_result = None

        if amount is not None:

            validation_result = validate_trip(
                employee_id=employee_id,
                trip_type=trip_type,
                amount=amount,
                time=time,
            )

        # ----------------------------------------------------
        # Store tool result
        # ----------------------------------------------------

        state["tool_result"] = {
            "employee": employee_result,
            "trip_validation": validation_result,
        }

        # ----------------------------------------------------
        # RAG search
        # ----------------------------------------------------

        results = search_policy(
            query=query,
            model=self.embedding_model,
            index=self.index,
            chunks=self.chunks,
            top_k=TOP_K
        )

        state["policy_results"] = results

        # ----------------------------------------------------
        # No policy result
        # ----------------------------------------------------

        if not results:

            state["response"] = (
                "I found the employee information, "
                "but couldn't find enough policy "
                "information to answer the travel question."
            )

            state["sources"] = [
                "employee_eligibility.txt"
            ]

            self.save_memory(state)

            return state

        # ----------------------------------------------------
        # Build RAG context
        # ----------------------------------------------------

        context = build_context(results)

        # ----------------------------------------------------
        # Build employee/tool context
        # ----------------------------------------------------

        employee_context = f"""
Employee ID: {employee_id}
Country: {employee_result['country']}
Employee Type: {employee_result['employee_type']}
Eligibility Status: {employee_result['status']}
Manager Approval: {employee_result['manager_approval']}
"""

        # ----------------------------------------------------
        # Add trip validation information
        # ----------------------------------------------------

        if validation_result:

            employee_context += f"""
Trip Type: {trip_type}
Trip Amount: {amount}
Trip Time: {time or "Not specified"}

Trip Validation:
{validation_result}
"""

        else:

            employee_context += f"""
Trip Type: {trip_type}
Trip Amount: Not specified
Trip Time: {time or "Not specified"}
"""

        # ----------------------------------------------------
        # Gemini combined prompt
        # ----------------------------------------------------

        prompt = f"""
You are a corporate travel policy assistant.

Answer the employee's question using BOTH:

1. Employee/trip information returned by tools
2. Company policy retrieved through RAG

Rules:
- Do not invent policy information.
- Do not invent approval records.
- "Approval Required" does NOT mean approved.
- Do not claim additional approval exists unless
  explicitly provided.
- Use the employee's actual country.
- Preserve the correct currency.
- If information is missing, clearly say what is missing.
- Keep the answer concise and practical.
- Mention the relevant policy source.
- Do not confuse manager approval with additional approval.
- If the trip amount exceeds the policy limit, clearly
  explain the reimbursable limit and excess amount.

EMPLOYEE AND TRIP INFORMATION:
{employee_context}

RETRIEVED POLICY CONTEXT:
{context}

EMPLOYEE QUESTION:
{query}

FINAL ANSWER:
"""

        # ----------------------------------------------------
        # Gemini call
        # ----------------------------------------------------

        response = self.llm.invoke(prompt)

        state["response"] = (
            self.extract_response_text(response)
        )

        # ----------------------------------------------------
        # Sources
        # ----------------------------------------------------

        state["sources"] = (
            self.unique_sources(results)
        )

        if (
            "employee_eligibility.txt"
            not in state["sources"]
        ):

            state["sources"].append(
                "employee_eligibility.txt"
            )

        self.save_memory(state)

        return state

    # ========================================================
    # SAVE MEMORY
    # ========================================================

    def save_memory(
        self,
        state: AgentState
    ):

        query = state["query"]

        # ----------------------------------------------------
        # Save conversation
        # ----------------------------------------------------

        self.memory.add_message(
            "user",
            query
        )

        if state.get("response"):

            self.memory.add_message(
                "assistant",
                state["response"]
            )

        # ----------------------------------------------------
        # Save travel context
        # ----------------------------------------------------

        self.memory.update_context(

            employee_id=state.get(
                "employee_id"
            ),

            trip_type=state.get(
                "trip_type"
            ),

            amount=state.get(
                "amount"
            ),

            time=state.get(
                "time"
            ),
        )

        # ----------------------------------------------------
        # Save country
        # ----------------------------------------------------

        tool_result = state.get(
            "tool_result"
        )

        if not tool_result:
            return

        employee_result = (
            tool_result.get(
                "employee"
            )
        )

        if employee_result:

            self.memory.update_context(
                country=employee_result.get(
                    "country"
                )
            )

        elif tool_result.get("country"):

            self.memory.update_context(
                country=tool_result.get(
                    "country"
                )
            )

    # ========================================================
    # CLEAR MEMORY
    # ========================================================

    def clear_memory(self):

        self.memory.clear()

        print(
            "Conversation memory cleared."
        )

    # ========================================================
    # GET MEMORY
    # ========================================================

    def get_memory(self):

        return {
            "history":
                self.memory.get_history(),

            "context":
                self.memory.get_context(),
        }

    # ========================================================
    # EXTRACT AMOUNT
    # ========================================================

    @staticmethod
    def extract_amount(
        query: str
    ) -> float | None:

        patterns = [

            r"₹\s*([\d,]+(?:\.\d+)?)",

            r"INR\s*([\d,]+(?:\.\d+)?)",

            r"\$\s*([\d,]+(?:\.\d+)?)",

            r"USD\s*([\d,]+(?:\.\d+)?)",

            (
                r"(?:amount|costs?|cost|price)"
                r"\s*(?:is|of|=)?"
                r"\s*₹?\$?"
                r"\s*([\d,]+(?:\.\d+)?)"
            ),

        ]

        for pattern in patterns:

            match = re.search(
                pattern,
                query,
                re.IGNORECASE
            )

            if match:

                value = (
                    match.group(1)
                    .replace(",", "")
                )

                try:

                    return float(value)

                except ValueError:

                    return None

        return None

    # ========================================================
    # EXTRACT TIME
    # ========================================================

    @staticmethod
    def extract_time(
        query: str
    ) -> str | None:

        patterns = [

            r"\b\d{1,2}:\d{2}\s*(?:AM|PM)?\b",

            r"\b\d{1,2}\s*(?:AM|PM)\b",

        ]

        for pattern in patterns:

            match = re.search(
                pattern,
                query,
                re.IGNORECASE
            )

            if match:

                return match.group(0)

        return None

    # ========================================================
    # EXTRACT TRIP TYPE
    # ========================================================

    @staticmethod
    def extract_trip_type(
        query: str
    ) -> str:

        query_lower = query.lower()

        if "airport" in query_lower:

            return "airport"

        if "customer" in query_lower:

            return "customer meeting"

        if "office" in query_lower:

            return "office"

        if "business" in query_lower:

            return "business"

        return "travel"

    # ========================================================
    # EXTRACT GEMINI RESPONSE
    # ========================================================

    @staticmethod
    def extract_response_text(
        response
    ) -> str:

        content = response.content

        # ----------------------------------------------------
        # Normal string response
        # ----------------------------------------------------

        if isinstance(
            content,
            str
        ):

            return content.strip()

        # ----------------------------------------------------
        # Gemini may return a list of content blocks
        # ----------------------------------------------------

        if isinstance(
            content,
            list
        ):

            text_parts = []

            for block in content:

                if isinstance(
                    block,
                    dict
                ):

                    text = block.get(
                        "text"
                    )

                    if text:

                        text_parts.append(
                            text
                        )

                elif isinstance(
                    block,
                    str
                ):

                    text_parts.append(
                        block
                    )

            return "\n".join(
                text_parts
            ).strip()

        return str(
            content
        ).strip()

    # ========================================================
    # UNIQUE SOURCES
    # ========================================================

    @staticmethod
    def unique_sources(
        results
    ) -> list[str]:

        sources = []

        for result in results:

            metadata = result.get(
                "metadata",
                {}
            )

            source = metadata.get(
                "source"
            )

            if source and source not in sources:

                sources.append(
                    source
                )

        return sources

    # ========================================================
    # PUBLIC RUN METHOD
    # ========================================================

    def run(
        self,
        query: str,
        employee_id: str | None = None
    ) -> dict[str, Any]:

        # ----------------------------------------------------
        # Empty question
        # ----------------------------------------------------

        if not query or not query.strip():

            return {

                "route": "error",

                "response":
                    "Please enter a travel "
                    "policy question.",

                "sources": [],

            }

        # ----------------------------------------------------
        # Initial state
        # ----------------------------------------------------

        state: AgentState = {

            "query":
                query.strip(),

            "employee_id":
                employee_id,

        }

        # ----------------------------------------------------
        # Run LangGraph
        # ----------------------------------------------------

        result = self.graph.invoke(
            state
        )

        return result


# ============================================================
# TESTING
# ============================================================

if __name__ == "__main__":

    agent = TravelPolicyAgent()

    # ========================================================
    # TEST 1: GENERAL POLICY QUESTION
    # ========================================================

    print("\n")
    print("=" * 70)
    print("GENERAL POLICY TEST")
    print("=" * 70)

    question = (
        "What is the airport reimbursement limit in India?"
    )

    print("\nUSER:")
    print(question)

    result = agent.run(
        question
    )

    print("\nROUTE:")
    print(
        result.get("route")
    )

    print("\nASSISTANT:")
    print(
        result.get("response")
    )

    print("\nSOURCES:")
    print(
        result.get("sources")
    )

    # ========================================================
    # TEST 2: EMPLOYEE + TRIP
    # ========================================================

    print("\n")
    print("=" * 70)
    print("EMPLOYEE TRIP TEST")
    print("=" * 70)

    question_1 = (
        "Can EMP001 take an airport trip "
        "costing ₹1,500?"
    )

    print("\nUSER:")
    print(question_1)

    result_1 = agent.run(
        question_1
    )

    print("\nROUTE:")
    print(
        result_1.get("route")
    )

    print("\nASSISTANT:")
    print(
        result_1.get("response")
    )

    print("\nSOURCES:")
    print(
        result_1.get("sources")
    )

    # ========================================================
    # TEST 3: MEMORY FOLLOW-UP
    # ========================================================

    print("\n")
    print("=" * 70)
    print("MEMORY FOLLOW-UP TEST")
    print("=" * 70)

    question_2 = (
        "What if it costs ₹2,500?"
    )

    print("\nUSER:")
    print(question_2)

    result_2 = agent.run(
        question_2
    )

    print("\nROUTE:")
    print(
        result_2.get("route")
    )

    print("\nASSISTANT:")
    print(
        result_2.get("response")
    )

    print("\nSOURCES:")
    print(
        result_2.get("sources")
    )

    # ========================================================
    # MEMORY
    # ========================================================

    print("\n")
    print("=" * 70)
    print("CURRENT MEMORY")
    print("=" * 70)

    memory = agent.get_memory()

    print("\nCONTEXT:")

    print(
        memory["context"]
    )

    print("\nHISTORY:")

    for message in memory["history"]:

        print(
            f"{message['role'].upper()}: "
            f"{message['message']}"
        )

