# Echo AI 🤖 Multi-Agent AI Framework

A powerful, plugin-based multi-agent conversational AI framework built on FastAPI and LangGraph, featuring intelligent
agent orchestration, efficient conversation storage, and comprehensive multi-provider LLM support.

## 🚀 Quick Start

### For End Users

```bash
# Install and start everything
pip install echo-app
python -m echo start all
```

### For Developers

```bash
# Clone and setup
git clone <your-repo-url>
cd echo
poetry install
python -m echo start all --debug
```

## 📋 Table of Contents

- [Features](#-features)
- [Installation](#-installation)
- [Usage](#-usage)
- [API Documentation](#-api-documentation)
- [Plugin Development](#-plugin-development)
- [Configuration](#-configuration)
- [Development](#-development)
- [Deployment](#-deployment)
- [Architecture](#-architecture)

## 🌟 Features

### 🤖 Multi-Agent Orchestration

- **LangGraph-Based Coordination**: Intelligent conversation routing between specialized agents
- **Plugin System**: SDK-based agent discovery with hot reload capabilities
- **Safety Mechanisms**: Configurable limits for agent hops and tool calls
- **State Management**: Comprehensive conversation state tracking with checkpoint persistence

### 💾 Storage Architecture

- **Conversation Turns**: Stores user input and AI responses with context
- **Multi-Backend Support**: PostgreSQL, Redis, and in-memory options
- **Token Tracking**: Precise cost attribution and optimization

### 🧠 Multi-Provider LLM Support

- **OpenAI**: GPT models with function calling
- **Anthropic**: Claude models with large context windows
- **Google**: Gemini models with multimodal capabilities
- **Azure OpenAI**: Enterprise-grade hosted models

### 🖥️ Command Line Interface

- **Simple & Reliable**: Built with argparse for maximum compatibility
- **Service Orchestration**: Start multiple services with a single command
- **Development Tools**: Debug mode, custom ports, dependency checking

## 📦 Installation

### Prerequisites

- **Python 3.13+**
- **Poetry** (for development)
- **PostgreSQL** (optional, for persistent storage)
- **Redis** (optional, for session storage)

### From PyPI (End Users)

```bash
pip install echo-app
```

### From Source (Developers)

```bash
git clone <your-repo-url>
cd echo
poetry install
```

## 🎯 Usage

### Starting Services

```bash
# Start both API and UI servers
python -m echo start all

# Start only the API server
python -m echo start api

# Start only the UI server
python -m echo start ui

# Start with custom configuration
python -m echo start all --api-port 8080 --ui-port 8502

# Start with debug mode
python -m echo start all --debug
```

### CLI Commands

```bash
# Show version information
python -m echo version

# Show system information
python -m echo info

# List available plugins
python -m echo plugins

# Show help
python -m echo --help
python -m echo start --help
```

## 📖 API Documentation

### Interactive Docs

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

### Core Endpoints

#### Chat Processing

```bash
curl -X POST "http://localhost:8000/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Help me solve 2x + 5 = 15",
    "thread_id": "user-123-session",
    "user_id": "user-123"
  }'
```

#### Plugin Management

```bash
# List available plugins
curl "http://localhost:8000/api/v1/plugins"

# Get plugin details
curl "http://localhost:8000/api/v1/plugins/math_agent"

# Reload plugins (development)
curl -X POST "http://localhost:8000/api/v1/plugins/reload"
```

#### System Health

```bash
# Simple health check
curl "http://localhost:8000/api/v1/health"

# Detailed system status
curl "http://localhost:8000/api/v1/status"
```

## 🔌 Plugin Development

### Quick Plugin Example

Create a minimal plugin:

```python
# plugins/hello_agent/plugin.py
from echo_sdk.base.plugin import BasePlugin
from echo_sdk.base.metadata import PluginMetadata


class HelloPlugin(BasePlugin):
    @staticmethod
    def get_metadata() -> PluginMetadata:
        return PluginMetadata(
            name="hello_agent",
            version="0.1.0",
            description="Replies with a friendly greeting",
            capabilities=["greeting"],
        )

    @staticmethod
    def create_agent() -> BasePluginAgent:
        from .agent import HelloAgent
        return HelloAgent(HelloPlugin.get_metadata())
```

```python
# plugins/hello_agent/agent.py
from echo_sdk.base.agent import BasePluginAgent


class HelloAgent(BasePluginAgent):
    async def ainvoke(self, message: str) -> str:
        return f"Hello from {self.metadata.name}! You said: {message}"
```

### Plugin Structure

```
plugins/
└── your_plugin/
    ├── __init__.py
    ├── plugin.py      # Plugin contract and metadata
    ├── agent.py       # Agent implementation
    └── tools.py       # Tool implementations
```

### Docker Compose Setup

```yaml
services:
  echo:
    build: ..
    ports:
      - "8000:8000"
      - "8501:8501"
    environment:
      - ECHO_DEFAULT_LLM_PROVIDER=openai
      - ECHO_PLUGINS_DIR=["/usr/src/echo/plugins"]
    volumes:
      - ../plugins:/usr/src/echo/plugins:ro
```

**Note**: The Docker setup uses the new CLI commands via supervisord to manage both API and UI services.

## ⚙️ Configuration

### Environment Variables

```bash
# LLM Provider
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

### CLI Override

Most settings can be overridden via CLI arguments:

```bash
python -m echo start api --api-host 127.0.0.1 --api-port 8080 --debug
```

### Safety and Performance

```bash
# Agent Routing Limits
ECHO_MAX_AGENT_HOPS=25
ECHO_MAX_TOOL_HOPS=50
ECHO_GRAPH_RECURSION_LIMIT=50

# Session Management
ECHO_SESSION_TIMEOUT=3600
ECHO_MAX_SESSION_HISTORY=100
```

## 🔧 Development

### Setup

```bash
# Install development dependencies
poetry install

# Run with auto-reload using CLI (recommended)
python -m echo start all --debug
```

### Development Commands

```bash
# Run tests
poetry run pytest

# Code formatting
poetry run black src/
poetry run isort src/

# Show detailed system information
echo info

# List and manage plugins
echo plugins
```

### Project Structure

```
echo/
├── src/echo/                    # Main application code
│   ├── api/                     # FastAPI routes and schemas
│   ├── cli.py                   # Command-line interface (argparse)
│   ├── core/                    # Multi-agent orchestration
│   ├── domain/                  # Business models and DTOs
│   ├── infrastructure/          # External service integrations
│   ├── config/                  # Configuration management
│   └── ui/                      # Streamlit chat interface
├── plugins/                     # Plugin ecosystem
├── sdk/                         # Echo SDK for plugin development
├── docs/                        # Documentation
├── migrations/                  # Database migrations
├── deploy.sh                    # PyPI deployment script
└── tests/                       # Test suite
```

## 🚀 Deployment

### Health Checks

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

RUN pip install poetry && poetry install
EXPOSE 8000 8501

# Use the CLI for better control
CMD ["poetry", "run", "echo", "start", "all"]
```

### PyPI Deployment (Maintainers)

```bash
# Deploy with patch version bump
./deploy.sh --patch

# Deploy with minor version bump
./deploy.sh --minor

# Deploy to TestPyPI first
./deploy.sh --test

# Deploy without version bump
./deploy.sh --no-bump
```

## 🏗️ Architecture

### Framework Architecture

```
Echo AI Framework Architecture
├── CLI Layer (argparse)
│   ├── Service Management (start/stop api, ui, or both)
│   ├── System Information (version, info, plugins)
│   └── Development Tools (debug mode, custom config)
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

### Storage Architecture

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

**Echo AI Framework** - Empowering intelligent multi-agent conversations with production-ready performance and
developer-friendly architecture.
