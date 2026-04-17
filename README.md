conda create -n doc-code python=3.13 -y

conda activate doc-code

pip install -r requirements.txt

streamlit run streamlit_app.py

1. Overall Execution Flow
main.py
   │
   ▼
OrchestratorAgent
   │
   ├───────────────┬──────────────────┐
   ▼               ▼                  ▼
EngineeringAgent  GraphAgent         Tool Agents
(RAG Search)      (Graph-RAG)        (Weather / Tender / APIs)
   │               │
   ▼               ▼
Retriever       GraphQuery
   │               │
   ▼               ▼
Vector Store     GraphStore
(Azure Search /
 FAISS)
   │
   ▼
LLM (Azure OpenAI / OpenAI)
   │
   ▼
Final Answer

2️⃣ File-Level Call Hierarchy

main.py
   │
   ▼
agents/orchestrator_agent.py
   │
   ├── calls → agents/engineering_agent.py
   │             │
   │             ▼
   │        retrieval/retriever.py
   │             │
   │             ▼
   │        vectorstore/vector_store.py
   │             │
   │             ▼
   │        Azure AI Search / FAISS
   │
   ├── calls → agents/graph_agent.py
   │             │
   │             ▼
   │        graph_rag/graph_query.py
   │             │
   │             ▼
   │        graph_rag/graph_store.py
   │
   └── calls → agents/tool_agents
                 │
                 ├── tools/weather_tool.py
                 ├── tools/tender_tool.py
                 └── tools/project_tool.py

3️⃣ Document Ingestion Pipeline

index_documents.py (This does not exist. It is actually Main.py)
   │
   ▼
document_ingestion/document_processor.py
   │
   ▼
document_ingestion/hierarchical_chunker.py
   │
   ▼
vectorstore/embedding_service.py
   │
   ▼
vectorstore/vector_store.py
   │
   ▼
Azure AI Search / FAISS

Document
   │
   ▼
GraphBuilder
   │
   ▼
GraphExtractor
   │
   ▼
GraphStore


4️⃣ Complete Project Structure With Responsibilities

src
│
├── main.py
│       Application entry point
│
├── config
│       config.py
│       Loads LLM, chunk size, environment settings
│
├── agents
│   │
│   ├── orchestrator_agent.py
│   │       Decides which agent should answer
│   │
│   ├── engineering_agent.py
│   │       Uses RAG (vector search)
│   │
│   ├── graph_agent.py
│   │       Queries knowledge graph
│   │
│   └── tool_agents
│           weather_agent.py
│           tender_agent.py
│
├── document_ingestion
│   │
│   ├── document_processor.py
│   │       Loads PDFs
│   │
│   └── hierarchical_chunker.py
│           Section / Clause chunking
│
├── retrieval
│   │
│   └── retriever.py
│           Retrieves documents from vector store
│
├── vectorstore
│   │
│   ├── vector_store.py
│   │       FAISS / Azure Search connection
│   │
│   └── embedding_service.py
│           Creates embeddings
│
├── graph_rag
│   │
│   ├── graph_extractor.py
│   │       LLM extracts entities
│   │
│   ├── graph_builder.py
│   │       Builds graph
│   │
│   ├── graph_store.py
│   │       Stores graph (networkx)
│   │
│   └── graph_query.py
│           Query graph
│
├── tools
│       weather_tool.py
│       tender_tool.py
│
└── debug
        rag_debugger.py

5️⃣ Example Execution (Real Scenario)

user asks
Which bridges in Maharashtra used M40 concrete?

Flow:
User Question
   │
   ▼
main.py
   │
   ▼
OrchestratorAgent
   │
   ▼
GraphAgent
   │
   ▼
graph_query.py
   │
   ▼
graph_store.py
   │
   ▼
Return project list
   │
   ▼
LLM formats answer

6️⃣ Another Example
User asks:
What is the drainage specification for highways?

Flow:
User Question
   │
   ▼
OrchestratorAgent
   │
   ▼
EngineeringAgent
   │
   ▼
Retriever
   │
   ▼
Vector Store
   │
   ▼
Relevant document chunks
   │
   ▼
LLM
   │
   ▼
Answer

three independent layers:
Layer 1
Data Ingestion
(document_processor)

Layer 2
Knowledge Storage
(vectorstore + graph)

Layer 3
Agents
(orchestrator + engineering + graph)

Microsoft-Style Azure AI Assistant Architecture

                         ┌───────────────────────────┐
                         │        Users               │
                         │ Web / App / Teams / API   │
                         └──────────────┬────────────┘
                                        │
                                        ▼
                         ┌───────────────────────────┐
                         │        Application        │
                         │  Backend API / Agent      │
                         │  (Python / FastAPI)       │
                         └──────────────┬────────────┘
                                        │
                       ┌────────────────┴────────────────┐
                       │                                 │
                       ▼                                 ▼
             ┌───────────────────┐             ┌───────────────────┐
             │     Tools Layer   │             │   RAG Orchestrator │
             │ Weather / Tender  │             │ Agent Routing      │
             │ External APIs     │             │ Query Planning     │
             └──────────┬────────┘             └──────────┬────────┘
                        │                                 │
                        ▼                                 ▼
               ┌────────────────────────────────────────────────┐
               │                 Retrieval Layer                 │
               │                                                │
               │   ┌───────────────────────────┐                │
               │   │      Azure AI Search      │                │
               │   │  Vector + Hybrid Search   │                │
               │   │  Engineering Documents    │                │
               │   └──────────────┬────────────┘                │
               │                  │                             │
               │                  ▼                             │
               │        ┌───────────────────┐                   │
               │        │  Graph-RAG Layer  │                   │
               │        │  Project Graph    │                   │
               │        │  (networkx/Neo4j) │                   │
               │        └───────────────────┘                   │
               └──────────────────────┬────────────────────────┘
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │      Azure OpenAI /       │
                        │      Azure AI Foundry     │
                        │   GPT-4o / GPT-4.1 / o1   │
                        └──────────────┬────────────┘
                                       │
                                       ▼
                          ┌────────────────────────┐
                          │      Final Answer      │
                          │  Citations + Sources   │
                          └────────────────────────┘


Data Ingestion Architecture (Another Microsoft Diagram)

          ┌────────────────────────────┐
          │   Enterprise Data Sources  │
          │                            │
          │ SharePoint                 │
          │ PDF Documents              │
          │ Engineering Specs          │
          │ Contracts                  │
          └─────────────┬──────────────┘
                        │
                        ▼
          ┌────────────────────────────┐
          │     Ingestion Pipeline     │
          │  Azure Function / Python   │
          └─────────────┬──────────────┘
                        │
                        ▼
          ┌────────────────────────────┐
          │      Document Processing   │
          │                            │
          │  Chunking                  │
          │  Metadata enrichment       │
          │  Section / Clause parsing  │
          └─────────────┬──────────────┘
                        │
                        ▼
          ┌────────────────────────────┐
          │      Embedding Model       │
          │  Azure OpenAI Embeddings   │
          └─────────────┬──────────────┘
                        │
                        ▼
          ┌────────────────────────────┐
          │      Azure AI Search       │
          │  Vector Index + Metadata   │
          └─────────────┬──────────────┘
                        │
                        ▼
          ┌────────────────────────────┐
          │      Graph Extraction      │
          │  Entities / Relationships  │
          └─────────────┬──────────────┘
                        │
                        ▼
          ┌────────────────────────────┐
          │     Knowledge Graph        │
          │   (Graph-RAG Database)     │
          └────────────────────────────┘


3️⃣ How This Matches Your Current Project

Your code structure maps to this architecture:

![alt text](image.png)


User Question
   │
   ▼
OrchestratorAgent
   │
   ├── EngineeringAgent (RAG)
   │
   ├── GraphAgent (knowledge graph)
   │
   ├── WeatherAgent
   │
   ├── TenderAgent
   │
   └── ProjectAgent
