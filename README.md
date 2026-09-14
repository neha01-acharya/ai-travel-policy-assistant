# AI Travel & Policy Assistant

An AI-powered Travel & Policy Assistant that helps employees understand company travel policies, check travel eligibility, validate trip requests, and calculate reimbursable amounts.

The application combines **Retrieval-Augmented Generation (RAG), semantic search, employee tools, an AI agent, conversational memory, MCP, and a Flask web interface** to provide policy-grounded answers and employee-specific travel assistance.

## 1. Business Problem

Employees often need quick answers to questions such as:

* What is the airport travel reimbursement limit?
* Am I eligible for a company-sponsored trip?
* Does my trip require additional approval?
* How much of my trip amount is reimbursable?
* Are late-night airport trips allowed?
* What happens if my trip exceeds the country limit?

Manually searching policy documents can be time-consuming and may lead to inconsistent interpretations.

The goal of this project is to provide a conversational assistant that can understand natural-language questions and use the **right source for the right question**:

* Policy questions → RAG
* Employee-specific questions → Employee tools
* Questions requiring both policy and employee information → Combined RAG + tools

---

## 2. Objectives

The assistant is designed to:

1. Answer questions from company travel policies.
2. Retrieve relevant policy information using semantic search.
3. Generate grounded answers using an LLM.
4. Provide policy source attribution.
5. Check employee eligibility.
6. Validate travel requests.
7. Calculate reimbursable amounts.
8. Handle follow-up questions using conversation memory.
9. Expose employee tools through MCP.
10. Provide a simple Flask-based web interface.
11. Handle invalid inputs and unsupported questions gracefully.

---

## 3. Key Features

### Policy Question Answering

The assistant can answer questions about:

* India travel policies
* US travel policies
* Airport travel
* Expense reimbursement
* Employee eligibility
* Cancellation
* Approval requirements

### Employee Eligibility

The assistant can check whether an employee is:

* Eligible
* Not Eligible
* Approval Required

### Trip Validation

The assistant validates:

* Employee ID
* Trip type
* Trip amount
* Travel time
* Country-specific limits
* Late-night travel conditions

### Reimbursement Calculation

The assistant calculates the reimbursable amount using the applicable policy limit.

For example:

```text
Trip amount: ₹2,500
India policy limit: ₹2,000

Reimbursable amount: ₹2,000
Excess amount: ₹500
Additional approval: Required
```

### Conversational Memory

The assistant maintains relevant information from the conversation so that follow-up questions can be understood.

Example:

```text
User:
Can EMP001 take an airport trip?

Assistant:
EMP001 is eligible for company-sponsored travel.

User:
What if it costs ₹2,500?

Assistant:
The India standard limit is ₹2,000. The additional ₹500 requires approval.
```

### Source Attribution

Policy answers include the policy documents used during retrieval, helping users understand where the answer came from.

---

# 4. System Architecture

```text
                         User
                           |
                           v
                    Flask Web UI
                           |
                           v
                    Travel Policy Agent
                           |
             +-------------+-------------+
             |                           |
             v                           v
        Intent / Route              Conversation
          Decision                    Memory
             |
       +-----+------+
       |            |
       v            v
      RAG          Tools
       |            |
       v            v
  Policy KB     Employee CSV
       |
       v
 Embeddings
       |
       v
     FAISS
       |
       v
  Relevant Chunks
       |
       v
      LLM
       |
       v
    Response
```

The agent determines whether a question should be answered using:

```text
RAG
Tool
RAG + Tool
```

This follows the project's **"right question → right source"** approach.

---

# 5. Technology Stack

| Component            | Technology                 |
| -------------------- | -------------------------- |
| Programming Language | Python                     |
| Data Processing      | Pandas                     |
| LLM                  | Google Gemini              |
| Embeddings           | Sentence Transformers      |
| Embedding Model      | all-MiniLM-L6-v2           |
| Vector Database      | FAISS                      |
| RAG                  | Custom Python RAG pipeline |
| Agent                | Custom routing agent       |
| Memory               | Conversation memory        |
| MCP                  | Model Context Protocol     |
| Backend              | Flask                      |
| Frontend             | HTML, CSS, JavaScript      |
| Version Control      | Git & GitHub               |

---

# 6. Policy Knowledge Base

The application uses seven fictional company policy documents:

```text
data/company_policy/
├── travel_policy_india.txt
├── travel_policy_us.txt
├── airport_policy.txt
├── employee_eligibility.txt
├── expense_policy.txt
├── cancellation_policy.txt
└── approval_policy.txt
```

Each document contains policy information relevant to corporate travel and employee operations.

---

# 7. Document Ingestion

The ingestion pipeline:

1. Reads all `.txt` policy files.
2. Handles missing or empty files.
3. Cleans unnecessary whitespace and formatting.
4. Preserves policy headings and important information.
5. Creates metadata for each document.
6. Splits documents into smaller chunks.
7. Saves the processed chunks.

Example metadata:

```json
{
  "source": "travel_policy_india.txt",
  "policy_type": "travel",
  "country": "India"
}
```

The ingestion process currently processes:

```text
Policy documents: 7
Total characters: ~10,115
Chunk size: 500
Chunk overlap: 100
Total chunks: 28
```

---

# 8. Chunking Experiment

Different chunk sizes were tested to understand their effect on retrieval.

| Chunk Size | Overlap | Number of Chunks |
| ---------: | ------: | ---------------: |
|        300 |      50 |               49 |
|        500 |     100 |               28 |
|        800 |     150 |               18 |

A smaller chunk size creates more focused pieces of information but produces more chunks.

A larger chunk size produces fewer chunks but may include unrelated information.

For this project, **500 characters with 100 characters overlap** was selected as a practical balance between retrieval precision and context.

---

# 9. RAG Pipeline

The Retrieval-Augmented Generation pipeline works as follows:

```text
User Question
      |
      v
Query Embedding
      |
      v
FAISS Semantic Search
      |
      v
Top-K Relevant Chunks
      |
      v
Policy Context
      |
      v
Prompt + Context
      |
      v
Gemini LLM
      |
      v
Grounded Answer
      |
      v
Policy Sources
```

The RAG implementation is located in:

```text
src/rag.py
```

The application retrieves the most relevant policy chunks before generating the answer.

The current configuration uses:

```text
Top K = 3
Similarity threshold = 0.40
```

---

# 10. Embeddings & Semantic Search

Policy chunks are converted into numerical vectors using:

```text
all-MiniLM-L6-v2
```

The model produces:

```text
384-dimensional embeddings
```

The embeddings are normalized and stored locally for semantic retrieval.

FAISS is used to search for chunks that are semantically similar to the user's question.

This allows the assistant to understand related questions even when the user's wording does not exactly match the policy text.

Example:

```text
User:
How much can I claim for an airport ride in India?

Policy:
India standard airport trip limit is INR 2,000 per trip.
```

The semantic search can identify the relevant policy even though the wording is different.

---

# 11. Agent Routing

The Travel Policy Agent decides which source should handle the question.

### Route 1 — RAG

Used for general policy questions.

Example:

```text
What is the airport reimbursement limit in India?
```

Flow:

```text
Question → RAG → Policy Context → LLM → Answer
```

### Route 2 — Tool

Used for employee-specific questions.

Example:

```text
Is EMP001 eligible for travel?
```

Flow:

```text
Question → Employee Tool → Employee Data → Answer
```

### Route 3 — Combined

Used when both employee information and policy information are required.

Example:

```text
Can EMP001 take an airport trip costing ₹2,500?
```

Flow:

```text
Question
   |
   +----> Employee Tool
   |
   +----> Policy Retrieval
             |
             v
       Combined Context
             |
             v
            LLM
             |
             v
           Answer
```

---

# 12. Employee Tools

The main employee tools are implemented in:

```text
src/tools.py
```

### `check_employee_eligibility()`

Checks an employee's:

* Employee ID
* Country
* Employee type
* Eligibility status
* Manager approval

### `validate_trip()`

Validates:

* Employee eligibility
* Trip type
* Trip amount
* Country limit
* Travel time
* Late-night conditions

### `calculate_reimbursement()`

Calculates:

* Reimbursable amount
* Excess amount
* Whether additional approval is required

---

# 13. Employee Dataset

The project contains a fictional employee dataset:

```text
data/employees.csv
```

Example records:

| Employee | Country       | Type       | Status            |
| -------- | ------------- | ---------- | ----------------- |
| EMP001   | India         | Full-Time  | Eligible          |
| EMP002   | India         | Contractor | Approval Required |
| EMP003   | United States | Full-Time  | Eligible          |
| EMP004   | United States | Contractor | Not Eligible      |
| EMP005   | India         | Full-Time  | Eligible          |
| EMP006   | United States | Full-Time  | Approval Required |

Unknown employees are not assumed to be eligible.

---

# 14. Country Limits

The fictional policies define the following standard limits:

| Country       | Standard Trip Limit |
| ------------- | ------------------: |
| India         |              ₹2,000 |
| United States |                 $75 |

Trips above the standard limit require additional approval before the excess can be considered reimbursable.

---

# 15. Late-Night Travel

The policy supports late-night airport trips between:

```text
10:00 PM – 6:00 AM
```

Such trips can be permitted when the required business conditions and approvals are satisfied.

The assistant does not treat late-night travel as automatically approved.

---

# 16. Prompt Engineering

The project applies the **P.T.C.F. framework**:

```text
P → Persona
T → Task
C → Context
F → Format
```

The prompts define:

* The assistant's role
* The task to perform
* Retrieved policy context
* Response format
* Grounding constraints

The assistant is instructed to:

* Use the provided policy context.
* Avoid inventing policy information.
* Clearly state when information is unavailable.
* Provide supporting policy sources.

Prompt examples and documentation are available in:

```text
docs/prompt_engineering.md
```

---

# 17. Hallucination Prevention

The assistant is designed to avoid making up information that does not exist in the policy documents.

Example:

```text
User:
What is the hotel reimbursement amount?

Assistant:
The provided policy context does not specify a hotel reimbursement amount.
```

Similarly, the assistant should not invent:

* Maximum flight ticket prices
* Rental car limits
* Cancellation fee amounts
* Approval records

when those details are not present in the knowledge base.

---

# 18. MCP Integration

The project includes an MCP server:

```text
mcp/server.py
```

The MCP server exposes the employee tools:

```text
check_employee_eligibility
validate_trip
calculate_reimbursement
```

This allows the tools to be exposed through the **Model Context Protocol**.

---

# 19. Flask Web Application

The Flask application is located at:

```text
app/app.py
```

The frontend contains:

```text
app/
├── app.py
├── templates/
│   └── index.html
└── static/
    ├── css/
    │   └── style.css
    └── js/
        └── app.js
```

The UI provides:

* Employee ID input
* Natural-language question input
* Assistant response
* Conversation interaction
* Policy sources
* Quick questions
* Clear conversation functionality
* Error messages

---

# 20. API Routes

### `GET /`

Loads the web interface.

### `POST /ask`

Processes an employee question.

Example request:

```json
{
  "employee_id": "EMP001",
  "question": "What is the airport reimbursement limit?"
}
```

### `POST /clear`

Clears the conversation memory.

### `GET /health`

Returns the application health status.

Example:

```json
{
  "status": "healthy",
  "service": "AI Travel Policy Assistant"
}
```

---

# 21. Project Structure

```text
ai-travel-policy-assistant/
│
├── app/
│   ├── app.py
│   ├── templates/
│   │   └── index.html
│   └── static/
│       ├── css/
│       │   └── style.css
│       └── js/
│           └── app.js
│
├── data/
│   ├── company_policy/
│   │   ├── travel_policy_india.txt
│   │   ├── travel_policy_us.txt
│   │   ├── airport_policy.txt
│   │   ├── employee_eligibility.txt
│   │   ├── expense_policy.txt
│   │   ├── cancellation_policy.txt
│   │   └── approval_policy.txt
│   │
│   ├── employees.csv
│   └── processed/
│       └── chunks.json
│
├── docs/
│   └── prompt_engineering.md
│
├── mcp/
│   └── server.py
│
├── src/
│   ├── agent.py
│   ├── embeddings.py
│   ├── ingestion.py
│   ├── memory.py
│   ├── prompts.py
│   ├── rag.py
│   ├── tools.py
│   └── chunk_experiment.py
│
├── tests/
│   ├── test_cases.md
│   └── test_mcp.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 22. Testing

The project includes testing for:

### Policy Retrieval

* India airport reimbursement limit
* US airport reimbursement limit
* Late-night airport travel
* Approval requirements
* Unsupported policy questions

### Employee Eligibility

* EMP001
* EMP002
* EMP003
* EMP004
* EMP005
* EMP006
* Unknown employee IDs

### Trip Validation

Examples tested include:

```text
EMP001 + ₹1,500 airport trip
EMP001 + ₹2,500 airport trip
EMP003 + $50 airport trip
EMP003 + $100 airport trip
EMP001 + late-night airport trip
```

### Reimbursement

Examples include:

```text
₹1,500 / ₹2,000 → ₹1,500 reimbursable

₹2,000 / ₹2,000 → ₹2,000 reimbursable

₹2,500 / ₹2,000 → ₹2,000 reimbursable
                       ₹500 excess
                       Approval required

$50 / $75 → $50 reimbursable

$100 / $75 → $75 reimbursable
             $25 excess
             Approval required
```

MCP tools were also tested for tool discovery and execution.

---

# 23. Error Handling

The application handles several failure scenarios:

* Empty questions
* Invalid employee IDs
* Missing employee information
* Invalid travel amounts
* Unsupported policy questions
* Missing policy information
* Retrieval failures
* Tool failures
* LLM/API failures

The assistant avoids assuming missing employee or policy information.

---

# 24. Running the Project

### Step 1 — Clone the repository

```bash
git clone https://github.com/neha01-acharya/ai-travel-policy-assistant.git
cd ai-travel-policy-assistant
```

### Step 2 — Create a virtual environment

```bash
python -m venv venv
```

Activate it:

Linux/macOS:

```bash
source venv/bin/activate
```

Windows:

```bash
venv\Scripts\activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Configure environment variables

Create a `.env` file and add the required LLM API configuration.

Example:

```text
GOOGLE_API_KEY=your_api_key
```

The `.env` file is intentionally excluded from Git.

### Step 5 — Run document ingestion

```bash
python src/ingestion.py
```

### Step 6 — Generate embeddings

```bash
python src/embeddings.py
```

### Step 7 — Run the Flask application

```bash
python app/app.py
```

Then open:

```text
http://127.0.0.1:5000
```

---

# 25. Example Questions

The assistant can be tested with questions such as:

```text
What is the airport reimbursement limit in India?

What is the airport reimbursement limit in the US?

Are late-night airport trips allowed?

Does a trip above the standard limit require approval?

Is EMP001 eligible for travel?

Is EMP002 eligible?

Can EMP001 take an airport trip?

Can EMP001 take an airport trip costing ₹2,500?

What would be reimbursed if the trip costs ₹1,500?

What happens if an airport trip costs ₹2,500?

What is the cancellation policy?

What information is required for reimbursement?
```

---

# 26. Design Principles

### Right Question → Right Source

One of the key design principles is choosing the appropriate source instead of sending every question through the same pipeline.

```text
Policy Question
      ↓
     RAG

Employee Question
      ↓
    Tool

Employee + Policy Question
      ↓
  RAG + Tool
```

### Grounded Responses

The assistant should prefer saying that information is unavailable rather than inventing unsupported policy details.

### Separation of Concerns

The project separates:

```text
UI
↓
Agent
↓
RAG / Tools / Memory
↓
Data / Vector Store / LLM
```

This makes the application easier to test, maintain, and extend.

---

# 27. Limitations

Current limitations include:

* Policy documents are fictional and limited in scope.
* Employee data is a small sample dataset.
* The application depends on LLM API availability and quota.
* The vector store is currently local.
* Authentication and role-based access control are not implemented.
* Persistent production-grade conversation storage is not implemented.
* Policy coverage is limited to the provided documents.
* MCP is implemented as a basic tool server rather than a production deployment.

---

# 28. Future Enhancements

Possible future improvements include:

* Support for additional countries.
* Document upload functionality.
* Persistent conversation history.
* Policy comparison across countries.
* Improved source highlighting.
* Automated RAG evaluation.
* Prompt-injection protection.
* User feedback and response rating.
* Structured JSON responses.
* Authentication and role-based access control.
* Docker deployment.
* REST API documentation.
* Production vector database.
* More advanced multi-agent workflows.

---

# 29. Conclusion

The **AI Travel & Policy Assistant** demonstrates how modern AI techniques can be combined with structured employee data to build a practical enterprise assistant.

The project brings together:

```text
Prompt Engineering
        +
Document Processing
        +
Embeddings
        +
FAISS Semantic Search
        +
RAG
        +
LLM
        +
Tool Calling
        +
Agent Routing
        +
Memory
        +
MCP
        +
Flask
```

The resulting system can answer policy questions, retrieve supporting information, perform employee-specific checks, validate travel requests, calculate reimbursements, and provide a conversational interface for employees.
