# Echo Framework Test Suite

This directory contains comprehensive test cases for the Echo Multi-agent AI Framework.

## Test Structure

```
tests/
├── __init__.py                 # Test package initialization
├── conftest.py                 # Pytest configuration and shared fixtures
├── run_tests.py               # Test runner script
├── README.md                  # This file
├── core/                      # Core module tests
│   └── orchestrator/
│       ├── __init__.py
│       ├── test_coordinator.py
│       └── test_state.py
├── config/                    # Configuration tests
│   ├── __init__.py
│   └── test_settings.py
├── api/                       # API tests
│   ├── __init__.py
│   └── test_chat.py
└── infrastructure/            # Infrastructure tests
    ├── __init__.py
    └── test_llm_factory.py
```

## Test Coverage

The test suite covers the following areas:

### Core Module Tests
- **Orchestrator Coordinator**: Tests for the multi-agent orchestration system
  - Tool logging handler functionality
  - Graph node management
  - Routing logic and decision making
  - State management and updates
  - Error handling and edge cases

- **State Management**: Tests for conversation state handling
  - AgentState creation and validation
  - State utility functions
  - State evolution through conversations
  - Plugin context management

### Configuration Tests
- **Settings**: Tests for application configuration
  - Default values validation
  - Environment variable loading
  - Field validation and constraints
  - Provider-specific configurations

### API Tests
- **Chat Router**: Tests for chat API endpoints
  - Conversation creation
  - Message sending and retrieval
  - Error handling and validation
  - Pagination and metadata handling

### Infrastructure Tests
- **LLM Factory**: Tests for language model factory
  - Provider-specific model creation
  - Configuration validation
  - Error handling

## Running Tests

### Prerequisites

Make sure you have the test dependencies installed:

```bash
pip install -r requirements-dev.txt
# or
poetry install --with dev
```

### Basic Test Execution

Run all tests:
```bash
python -m pytest
```

Run tests with coverage:
```bash
python -m pytest --cov=src/echo --cov-report=html
```

### Using the Test Runner Script

The `run_tests.py` script provides a convenient way to run tests:

```bash
# Run all tests
python tests/run_tests.py

# Run specific test file
python tests/run_tests.py tests/core/orchestrator/test_coordinator.py

# Run tests with specific markers
python tests/run_tests.py -m "unit"

# Run tests without coverage
python tests/run_tests.py --no-coverage

# List all available tests
python tests/run_tests.py --list-tests

# Verbose output
python tests/run_tests.py -v
```

### Test Markers

The test suite uses pytest markers for organization:

- `@pytest.mark.unit`: Unit tests
- `@pytest.mark.integration`: Integration tests
- `@pytest.mark.asyncio`: Async tests
- `@pytest.mark.slow`: Slow-running tests

Run tests by marker:
```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Run tests excluding slow ones
pytest -m "not slow"

# Run async tests
pytest -m asyncio
```

### Coverage Reports

Generate coverage reports:
```bash
# Terminal coverage report
pytest --cov=src/echo --cov-report=term-missing

# HTML coverage report
pytest --cov=src/echo --cov-report=html

# XML coverage report (for CI/CD)
pytest --cov=src/echo --cov-report=xml
```

## Test Fixtures

The test suite provides several shared fixtures in `conftest.py`:

### Core Fixtures
- `settings`: Test settings with safe defaults
- `mock_llm_factory`: Mocked LLM factory
- `mock_plugin_manager`: Mocked plugin manager
- `sample_agent_state`: Sample agent state for testing
- `empty_agent_state`: Empty agent state for testing

### Message Fixtures
- `sample_messages`: Sample LangChain messages
- `mock_tool_calls`: Mock tool call data
- `mock_tool_response`: Mock tool response data

### Environment Fixtures
- `test_env_vars`: Test environment variables
- `setup_test_env`: Automatic environment setup/teardown

## Writing New Tests

### Test File Structure

Follow this structure for new test files:

```python
"""Test cases for [module name]."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from echo.module.path import ClassToTest


class TestClassName:
    """Test cases for ClassName."""

    @pytest.fixture
    def instance(self):
        """Create test instance."""
        return ClassToTest()

    def test_method_name(self, instance):
        """Test specific method functionality."""
        # Arrange
        expected = "expected result"
        
        # Act
        result = instance.method()
        
        # Assert
        assert result == expected

    @pytest.mark.asyncio
    async def test_async_method(self, instance):
        """Test async method functionality."""
        # Arrange
        expected = "expected result"
        
        # Act
        result = await instance.async_method()
        
        # Assert
        assert result == expected
```

### Test Naming Conventions

- Test files: `test_*.py`
- Test classes: `Test*`
- Test methods: `test_*`
- Use descriptive names that explain what is being tested

### Test Organization

Organize tests using the AAA pattern:
- **Arrange**: Set up test data and conditions
- **Act**: Execute the method being tested
- **Assert**: Verify the results

### Mocking Guidelines

- Use `@patch` decorator for external dependencies
- Mock at the appropriate level (prefer higher-level mocks)
- Use `AsyncMock` for async methods
- Provide realistic mock return values

### Error Testing

Always test error conditions:
```python
def test_method_with_error(self, instance):
    """Test method error handling."""
    with pytest.raises(ValueError, match="Expected error message"):
        instance.method_that_raises_error()
```

## Continuous Integration

The test suite is configured for CI/CD with:

- Coverage reporting (minimum 80% required)
- Multiple output formats (terminal, HTML, XML)
- Integration with GitHub Actions
- Pre-commit hooks for test validation

## Debugging Tests

### Verbose Output
```bash
pytest -v -s
```

### Debug Specific Test
```bash
pytest tests/path/to/test_file.py::TestClass::test_method -v -s
```

### Run Tests with Debugger
```bash
pytest --pdb
```

### Generate Test Report
```bash
pytest --html=report.html --self-contained-html
```

## Best Practices

1. **Test Isolation**: Each test should be independent
2. **Descriptive Names**: Test names should clearly describe what is being tested
3. **Minimal Setup**: Keep test setup minimal and focused
4. **Mock External Dependencies**: Don't rely on external services in unit tests
5. **Test Edge Cases**: Include tests for boundary conditions and error cases
6. **Maintain Test Data**: Keep test data realistic and up-to-date
7. **Document Complex Tests**: Add comments for complex test scenarios

## Contributing

When adding new features or fixing bugs:

1. Write tests first (TDD approach)
2. Ensure all tests pass
3. Maintain or improve test coverage
4. Update this README if adding new test categories
5. Follow the existing test patterns and conventions

## Troubleshooting

### Common Issues

**Import Errors**: Make sure the Python path includes the `src` directory:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

**Missing Dependencies**: Install test dependencies:
```bash
pip install pytest pytest-asyncio pytest-cov pytest-mock
```

**Async Test Issues**: Use the `@pytest.mark.asyncio` decorator for async tests.

**Coverage Issues**: Ensure the coverage configuration includes the correct source paths.

For more help, check the pytest documentation or create an issue in the project repository.
