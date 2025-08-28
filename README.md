# LangGraph Multi-Agent Jeep Patriot Diagnostic Assistant

A sophisticated LangGraph-powered diagnostic assistant for 2011 Jeep Patriots that leverages semantic search and AI workflows to provide accurate diagnostics based exclusively on the official vehicle manual.

## 🚗 Overview

This project demonstrates advanced AI agent patterns using LangGraph for workflow orchestration, semantic search with ChromaDB, and OpenAI's GPT-4o-mini for intelligent automotive diagnostics. The system maintains strict adherence to official manual content, ensuring reliable and authoritative responses.

## ✨ Key Features

- **🔍 Semantic Search**: ChromaDB vector store with OpenAI embeddings for intelligent manual content retrieval
- **🔄 LangGraph Workflow**: State-driven workflow orchestration with type-safe state management
- **📚 Manual-First Approach**: Exclusively uses official 2011 Jeep Patriot manual content
- **🎯 Intelligent Query Analysis**: Analyzes user queries to identify relevant automotive systems
- **💬 Interactive CLI**: User-friendly command-line interface with comprehensive error handling
- **📊 Comprehensive Logging**: Detailed logging throughout the diagnostic process

## 🏗️ Architecture & Design Patterns

### State-Driven Workflow Pattern
The application uses LangGraph's `StateGraph` with a `TypedDict` for type-safe state management:

```python
class PatriotDiagnosticState(TypedDict):
    user_query: str
    relevant_sections: List[str]
    diagnosis: str
    recommendations: List[str]
    conversation_history: List[Dict]
```

### Node-Based Processing Pipeline
Sequential workflow nodes ensure systematic processing:
1. **`read_manual`** → Load and index PDF content
2. **`analyze_query`** → Parse user query and identify systems
3. **`search_manual`** → Semantic search for relevant sections
4. **`generate_diagnosis`** → AI-powered diagnosis generation
5. **`format_response`** → Final response formatting

### Semantic Search Integration
- **Vector Store**: ChromaDB with persistent storage
- **Embeddings**: OpenAI embeddings for semantic similarity
- **Chunking Strategy**: RecursiveCharacterTextSplitter (1000 chars, 200 overlap)
- **Search Methods**: Standard similarity search and score-based filtering

## 📁 Project Structure

```
langGraph-multi-agents/
├── main.py                    # CLI interface and application entry point
├── patriot_agent.py          # Core LangGraph workflow and diagnostic logic
├── semantic_pdf_reader.py    # PDF processing and semantic search capabilities
├── requirements.txt          # Python dependencies with pinned versions
├── .env.example             # Environment variable template
├── 2011-patriot manual.pdf # Official Jeep Patriot manual (required)
└── chroma_db/               # ChromaDB vector store (auto-generated)
```

## 🔧 Technical Stack

| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| **Workflow Engine** | LangGraph | 0.2.16 | State-driven workflow orchestration |
| **LLM Integration** | LangChain | 0.2.16 | LLM abstraction and prompt management |
| **Language Model** | OpenAI GPT-4o-mini | Latest | Query analysis and diagnosis generation |
| **Vector Database** | ChromaDB | 0.4.24 | Semantic search and document storage |
| **PDF Processing** | PyPDF | 6.0.0 | Document loading and text extraction |
| **Embeddings** | OpenAI Embeddings | Latest | Semantic similarity calculations |

## 🚀 Setup & Installation

### Prerequisites
- Python 3.8+
- OpenAI API key
- 2011 Jeep Patriot manual PDF

### Installation Steps

1. **Clone and Navigate**:
   ```bash
   git clone <repository-url>
   cd langGraph-multi-agents
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment**:
   ```bash
   cp .env.example .env
   # Edit .env and add your OpenAI API key
   echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
   ```

4. **Verify Manual Presence**:
   - Ensure `2011-patriot manual.pdf` is in the project root
   - The system will automatically index the manual on first run

## 💻 Usage

### Basic Usage
```bash
python main.py
```

### Example Interaction
```
🚗 Initializing Jeep Patriot Diagnostic Assistant...
📖 Loading manual content...
✅ Assistant ready!

============================================================
JEEP PATRIOT DIAGNOSTIC ASSISTANT
============================================================
Ask me about any issues with your 2011 Jeep Patriot.
I'll consult the official manual to help diagnose problems.
Type 'quit' or 'exit' to end the session.
============================================================

🔧 Describe your Patriot issue: Engine making knocking sounds during acceleration

🔍 Analyzing issue and consulting manual...

📋 DIAGNOSTIC RESPONSE:
----------------------------------------
Based on the manual content, engine knocking during acceleration 
typically indicates [detailed manual-based response]...
----------------------------------------
```

### Query Examples
- **Engine Issues**: "Engine making knocking sounds", "Engine won't start"
- **Transmission**: "Transmission slipping", "Hard shifting"
- **Electrical**: "Check engine light", "Battery not charging"
- **Brakes**: "Brakes feel spongy", "Brake pedal goes to floor"
- **Cooling**: "Engine overheating", "Coolant leak"

## 🔍 Code Architecture Deep Dive

### Main Application (`main.py`)
- **Responsibility**: CLI interface, error handling, user interaction loop
- **Key Features**: Environment validation, graceful error handling, formatted output
- **Design Pattern**: Command-line application with input validation

### Patriot Agent (`patriot_agent.py`)
- **Responsibility**: LangGraph workflow orchestration and diagnostic logic
- **Key Classes**: `PatriotAgent`, `PatriotDiagnosticState`
- **Design Patterns**: State machine, workflow orchestration, dependency injection
- **Workflow Nodes**:
  - `_read_manual_node()`: PDF loading and indexing
  - `_analyze_query_node()`: Query analysis with structured JSON output
  - `_search_manual_node()`: Semantic search with fallback strategies
  - `_generate_diagnosis_node()`: AI-powered diagnosis with manual constraints
  - `_format_response_node()`: Response formatting and presentation

### Semantic PDF Reader (`semantic_pdf_reader.py`)
- **Responsibility**: PDF processing, vector storage, semantic search
- **Key Classes**: `SemanticPatriotManualReader`
- **Design Patterns**: Repository pattern, factory pattern for vector store
- **Features**:
  - Persistent ChromaDB storage
  - Configurable chunking strategies
  - Score-based filtering
  - Metadata preservation (page numbers)

## 🎯 Key Design Decisions

### Manual-First Philosophy
- **Constraint**: Only uses official manual content, no general automotive knowledge
- **Benefit**: Ensures accuracy and reliability of diagnostic information
- **Implementation**: Explicit prompts constraining LLM responses to manual content

### Fresh Indexing Strategy
- **Decision**: Always recreates vector store on startup
- **Rationale**: Ensures proper indexing and prevents stale data issues
- **Trade-off**: Slightly longer startup time for guaranteed accuracy

### Error Handling Strategy
- **Comprehensive Logging**: All operations logged with appropriate levels
- **Graceful Degradation**: Fallback responses when search fails
- **User-Friendly Messages**: Clear error messages with actionable guidance

### State Management
- **TypedDict Usage**: Type-safe state definitions prevent runtime errors
- **Immutable Updates**: State updates return new state objects
- **Conversation History**: Maintains context for complex diagnostic sessions

## 🔧 Configuration Options

### Environment Variables
```bash
OPENAI_API_KEY=your_openai_api_key_here  # Required: OpenAI API access
```

### Customizable Parameters
- **Chunk Size**: Modify `chunk_size` in `semantic_pdf_reader.py` (default: 1000)
- **Chunk Overlap**: Adjust `chunk_overlap` for better context (default: 200)
- **Search Results**: Change `k` parameter for more/fewer results (default: 8)
- **Score Threshold**: Adjust relevance filtering in `search_with_score()`

## 🚨 Error Handling & Troubleshooting

### Common Issues
1. **Missing API Key**: Clear error message with setup instructions
2. **PDF Not Found**: File existence validation with helpful guidance
3. **ChromaDB Issues**: Automatic recreation of vector store
4. **Network Errors**: Graceful handling of OpenAI API failures

### Logging Configuration
```python
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

## 🔮 Extension Points

### Adding New Workflows
1. Define new node functions in `PatriotAgent`
2. Add nodes to workflow graph
3. Update state schema if needed
4. Connect nodes with appropriate edges

### Custom Search Strategies
1. Extend `SemanticPatriotManualReader`
2. Implement new search methods
3. Add configuration parameters
4. Update agent to use new strategies

### Additional Data Sources
1. Create new reader classes following the same pattern
2. Extend state schema for multiple sources
3. Add source selection logic to workflow

## 📊 Performance Considerations

- **Vector Store**: ChromaDB provides efficient similarity search
- **Chunking Strategy**: Balanced for context preservation and search accuracy
- **API Calls**: Minimized through intelligent caching and batching
- **Memory Usage**: Efficient document processing with streaming where possible

## 🤝 Contributing

When contributing to this project, please:
1. Follow the established architectural patterns
2. Maintain the manual-first philosophy
3. Add comprehensive logging for new features
4. Update documentation for any API changes
5. Ensure type safety with proper TypedDict usage

## 📄 License

This project is designed for educational and diagnostic assistance purposes. Ensure compliance with OpenAI's usage policies and any applicable automotive service regulations.
