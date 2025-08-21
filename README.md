# Echo 🤖 Multi-Agent AI Framework

A powerful, plugin-based multi-agent conversational AI framework built on FastAPI and LangGraph, featuring intelligent
agent orchestration, efficient conversation storage, and comprehensive multi-provider LLM support.

## 🌟 Key Features

### 🤖 **Multi-Agent Orchestration**

- **LangGraph-Based Coordination**: Intelligent conversation routing between specialized agents using a sequential,
  tool-routed graph architecture
- **Plugin System**: SDK-based agent discovery with hot reload capabilities and comprehensive health monitoring
- **Safety Mechanisms**: Configurable limits for agent hops and tool calls to prevent infinite loops
- **State Management**: Comprehensive conversation state tracking across agent switches with checkpoint persistence

### 💾 **Efficient Storage Architecture**

- **Conversation Turns**: Stores user input and AI responses with conversation context
- **Context Management**: Maintains conversation continuity for LangGraph processing
- **Token Tracking**: Precise cost attribution and optimization across all interactions
- **Multi-Backend Support**: PostgreSQL, Redis, and in-memory options with automatic connection management

### 🧠 **Multi-Provider LLM Support**

- **OpenAI**: GPT models with function calling capabilities
- **Anthropic**: Claude models with large context windows
- **Google**: Gemini models with multimodal capabilities
- **Azure OpenAI**: Enterprise-grade hosted OpenAI models
- **Intelligent Caching**: Model instance caching for performance optimization

### 🔧 **Production-Ready Architecture**

- **Layered Design**: Clean separation of Domain, Application, Infrastructure, and API layers
- **Health Monitoring**: Comprehensive system status and plugin health endpoints
- **Configuration Management**: Environment-based configuration with validation
- **Async/Await**: Non-blocking operations for high performance

## 🏗️ Architecture Overview

```
Echo Framework Architecture
├── API Layer (FastAPI)
│   ├── Chat Endpoints (/api/v1/chat)
│   ├── Plugin Management (/api/v1/plugins)  
│   └── System Monitoring (/api/v1/status, /api/v1/health)
├── Application Services
│   ├── ConversationService (Complete conversation lifecycle)
│   ├── OrchestratorService (LangGraph coordination wrapper)
│   └── ServiceContainer (Dependency injection and lifecycle)
├── Core Orchestration
│   ├── MultiAgentOrchestrator (LangGraph-based routing)
│   ├── AgentState (Conversation state management)
│   └── Coordinator (Tool-routed graph execution)
├── Domain Models
│   ├── Thread (Conversation containers with cost tracking)
│   ├── Conversation (User-assistant exchanges)
│   ├── User & Organization (Multi-tenant support)
│   └── DTOs (Data transfer objects for API)
└── Infrastructure
    ├── Database (Multi-backend: PostgreSQL/Redis/Memory)
    ├── LLM Factory (Multi-provider with caching)
    └── Plugin Manager (SDK-based agent discovery)
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.13+**
- **Poetry** (for dependency management)
- **PostgreSQL** (optional, for persistent storage)
- **Redis** (optional, for session storage and checkpoints)

### Installation

1. **Clone the repository**:

   ```bash
   git clone <your-repo-url>
   cd echo
   ```

2. **Install dependencies**:

   ```bash
   poetry install
   ```

3. **Configure environment**:

   ```bash
   cp env.sample .env
   # Edit .env with your configuration
   ```

4. **Set up database (optional)**:

   ```bash
   # For PostgreSQL support
   poetry run alembic upgrade head
   ```

### Basic Configuration

Edit `.env` file with your LLM provider credentials:

```bash
# LLM Provider (choose one or multiple)
ECHO_DEFAULT_LLM_PROVIDER=openai
ECHO_OPENAI_API_KEY=your-openai-key
ECHO_ANTHROPIC_API_KEY=your-claude-key
ECHO_GOOGLE_API_KEY=your-gemini-key

# Database (optional - defaults to in-memory)
ECHO_CONVERSATION_STORAGE_BACKEND=memory  # or postgresql

# Plugin Configuration
ECHO_PLUGINS_DIR=["./plugins/src/echo_plugins"]

# API Server
ECHO_API_HOST=0.0.0.0
ECHO_API_PORT=8000
ECHO_DEBUG=true
```

### Running the Framework

```bash
# Start the API server
poetry run python -m echo

# Or with custom configuration
ECHO_API_PORT=8080 poetry run python -m echo
```

The API will be available at: `http://localhost:8000`

## 📖 API Documentation

### Interactive API Docs

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

### Core Endpoints

#### **Chat Processing**

```bash
# Process a chat message
curl -X POST "http://localhost:8000/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Help me solve 2x + 5 = 15",
    "thread_id": "user-123-session",
    "user_id": "user-123"
  }'
```

#### **Plugin Management**

```bash
# List available plugins
curl "http://localhost:8000/api/v1/plugins"

# Get plugin details
curl "http://localhost:8000/api/v1/plugins/math_agent"

# Reload plugins (development)
curl -X POST "http://localhost:8000/api/v1/plugins/reload"
```

#### **System Health**

```bash
# Simple health check
curl "http://localhost:8000/api/v1/health"

# Detailed system status
curl "http://localhost:8000/api/v1/status"
```

## 🔌 Plugin Development

Echo supports a powerful plugin system for extending capabilities with specialized agents using the Echo SDK.

### Plugin Structure

```
plugins/
└── your_plugin/
    ├── __init__.py
    ├── plugin.py      # Plugin contract and metadata
    ├── agent.py       # Agent implementation
    └── tools.py       # Tool implementations
```

### Example Plugin

```python
# plugin.py
from echo_sdk import BasePlugin, PluginMetadata, BasePluginAgent


class MathPlugin(BasePlugin):
    @staticmethod
    def get_metadata() -> PluginMetadata:
        return PluginMetadata(
            name="mathematics",
            version="0.1.0",
            description="Mathematical calculations and arithmetic operations agent",
            capabilities=["addition", "subtraction", "multiplication", "division"],
            llm_requirements={
                "provider": "openai",
                "model": "gpt-4.1",
                "temperature": 0.2,
                "max_tokens": 1024,
            }
        )

    @staticmethod
    def create_agent() -> BasePluginAgent:
        from .agent import MathAgent
        return MathAgent(MathPlugin.get_metadata())
```

### Plugin Configuration

```bash
# Configure plugin directories in .env
ECHO_PLUGINS_DIR=["./plugins/src/echo_plugins", "./custom_plugins"]
```

## 🗄️ Database Backends

### Supported Backends

**Development/Testing:**

- **In-Memory**: Fast, no persistence required

**Production Options:**

- **PostgreSQL**: ACID compliance, complex queries, full-text search
- **Redis**: High-performance session storage with TTL and checkpoint persistence

### Configuration Examples

**PostgreSQL Setup:**

```bash
# .env configuration
ECHO_CONVERSATION_STORAGE_BACKEND=postgresql
ECHO_POSTGRES_URL=postgresql+asyncpg://user:pass@localhost/echo

# Run migrations
poetry run alembic upgrade head
```

**Redis Setup:**

```bash
# .env configuration
ECHO_REDIS_URL=redis://localhost:6379
ECHO_PERSISTENCE_CHECKPOINT_LAYER=redis
ECHO_PERSISTENCE_MEMORY_LAYER=redis
```

## 🔧 Development

### Development Setup

```bash
# Install development dependencies
poetry install --with dev

# Run with auto-reload
ECHO_DEBUG=true poetry run python -m echo

# Run tests
poetry run pytest

# Code formatting
poetry run black src/
poetry run isort src/
```

### Project Structure

```
echo/
├── src/echo/                    # Main application code
│   ├── api/                     # FastAPI routes and schemas
│   ├── core/                    # Multi-agent orchestration
│   ├── domain/                  # Business models and DTOs
│   ├── infrastructure/          # External service integrations
│   ├── config/                  # Configuration management
│   └── ui/                      # Streamlit chat interface
├── plugins/                     # Plugin ecosystem
├── sdk/                         # Echo SDK for plugin development
├── docs/                        # Documentation
├── migrations/                  # Database migrations
└── tests/                       # Test suite
```

## 📊 Storage Architecture

Echo implements an efficient conversation storage strategy designed for multi-agent workflows:

### Conversation Storage

```
📝 Stores: User Input → AI Response
💾 Storage: Conversation turns with context
🔄 Context: Full reconstruction capability for LangGraph
```

### Benefits

- **Efficient Storage**: Optimized conversation history management
- **Fast Loading**: Streamlined conversation history retrieval
- **Context Preservation**: Full LangGraph context reconstruction
- **Token Tracking**: Precise cost attribution and optimization

## ⚙️ Configuration Options

### LLM Provider Settings

```bash
# Provider Selection
ECHO_DEFAULT_LLM_PROVIDER=openai              # openai|anthropic|google|azure
ECHO_DEFAULT_LLM_TEMPERATURE=0.3              # Response creativity (0.0-1.0)
ECHO_DEFAULT_LLM_CONTEXT_WINDOW=32000         # Token limit

# OpenAI Configuration
ECHO_OPENAI_API_KEY=your-key
ECHO_OPENAI_DEFAULT_MODEL=gpt-4.1

# Anthropic Configuration  
ECHO_ANTHROPIC_API_KEY=your-key
ECHO_ANTHROPIC_DEFAULT_MODEL=claude-sonnet-4-20250514

# Google Configuration
ECHO_GOOGLE_API_KEY=your-key
ECHO_GEMINI_DEFAULT_MODEL=gemini-2.5-flash
```

### Safety and Performance

```bash
# Agent Routing Limits (prevents infinite loops)
ECHO_MAX_AGENT_HOPS=25                        # Max agent switches
ECHO_MAX_TOOL_HOPS=50                         # Max tool calls
ECHO_GRAPH_RECURSION_LIMIT=50                # LangGraph step limit

# Session Management
ECHO_SESSION_TIMEOUT=3600                     # Session timeout (seconds)
ECHO_MAX_SESSION_HISTORY=100                 # Max conversation turns

# Persistence Configuration
ECHO_PERSISTENCE_TYPE=checkpoint              # checkpoint|memory
ECHO_PERSISTENCE_CHECKPOINT_LAYER=redis       # redis|postgres|sqlite
ECHO_PERSISTENCE_MEMORY_LAYER=redis           # redis|postgres|sqlite
```

## 🎨 User Interface

Echo includes a built-in Streamlit chat interface for development and testing:

```bash
# Run the Streamlit UI
streamlit run src/echo/ui/app.py

# Or set custom backend URL
ECHO_API_BASE_URL=http://localhost:8000 streamlit run src/echo/ui/app.py
```

## 📈 Production Deployment

### Health Checks

Echo provides comprehensive health monitoring for production deployments:

```bash
# Load balancer health check
GET /api/v1/health
→ {"status": "healthy"}

# Detailed system status
GET /api/v1/status  
→ {
  "status": "operational",
  "available_plugins": ["math_agent", "search_agent"],
  "healthy_plugins": ["math_agent", "search_agent"], 
  "failed_plugins": [],
  "total_sessions": 42
}
```

### Docker Deployment

```dockerfile
FROM python:3.13-slim

WORKDIR /app
COPY . .

RUN pip install poetry && poetry install --without dev
EXPOSE 8000

CMD ["poetry", "run", "python", "-m", "echo"]
```

### Environment Variables

All configuration can be provided via environment variables with the `ECHO_` prefix:

- `ECHO_API_HOST` - API server host (default: 0.0.0.0)
- `ECHO_API_PORT` - API server port (default: 8000)
- `ECHO_DEBUG` - Debug mode (default: false)
- `ECHO_PLUGINS_DIR` - Plugin directories (default: ["./plugins/src/echo_plugins"])

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with proper documentation
4. Add tests for new functionality
5. Ensure all tests pass (`poetry run pytest`)
6. Run code formatting (`poetry run black src/ && poetry run isort src/`)
7. Commit your changes (`git commit -m 'Add amazing feature'`)
8. Push to the branch (`git push origin feature/amazing-feature`)
9. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **LangGraph**: For powerful multi-agent orchestration capabilities
- **FastAPI**: For high-performance async API framework
- **Echo SDK**: For comprehensive plugin development support
- **Pydantic**: For robust data validation and serialization
- **Streamlit**: For interactive development interface

---

**Echo Framework** - Empowering intelligent multi-agent conversations with production-ready performance and
developer-friendly architecture.

For detailed documentation, visit the `/docs` directory or check the inline API documentation at `/docs` when running
the server.
