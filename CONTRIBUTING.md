# Contributing to MedGen

Thank you for your interest in contributing to MedGen! This document provides guidelines and instructions for contributing.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment. Please:

- Be respectful of differing viewpoints and experiences
- Accept constructive criticism gracefully
- Focus on what is best for the community
- Show empathy towards other community members

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/CS3264.git
   cd CS3264
   ```
3. **Add the upstream remote**:
   ```bash
   git remote add upstream https://github.com/DESU-CLUB/CS3264.git
   ```

## Development Setup

### Backend Setup

```bash
# Install uv package manager
pip install uv

# Install dependencies
uv sync

# Copy environment file
cp .env.example .env
# Edit .env with your OpenAI API key

# Run the backend
uv run python backend.py
```

### Frontend Setup

```bash
cd frontend
npm install
npm start
```

### Running Tests

```bash
# Backend tests
uv run pytest

# Frontend tests
cd frontend && npm test
```

## Making Changes

### Branch Naming

Use descriptive branch names:
- `feature/add-new-model` - New features
- `fix/generation-bug` - Bug fixes
- `docs/update-readme` - Documentation updates
- `refactor/cleanup-api` - Code refactoring

### Commit Messages

Follow conventional commit format:

```
type(scope): short description

Longer description if needed.

Fixes #123
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Code style (formatting, semicolons, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

Examples:
```
feat(generation): add support for custom temperature range
fix(api): handle missing API key gracefully
docs(readme): add Docker instructions
```

## Pull Request Process

1. **Update your fork**:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature
   ```

3. **Make your changes** and commit them

4. **Push to your fork**:
   ```bash
   git push origin feature/your-feature
   ```

5. **Create a Pull Request** on GitHub

### PR Requirements

- [ ] Code follows the project's coding standards
- [ ] Tests pass locally
- [ ] New features include appropriate tests
- [ ] Documentation is updated if needed
- [ ] PR description explains the changes

## Coding Standards

### Python

- Follow [PEP 8](https://pep8.org/) style guide
- Use type hints for function parameters and return values
- Maximum line length: 100 characters
- Use docstrings for all public functions and classes

```python
def generate_synthetic_data(
    df: pd.DataFrame,
    n_samples: int,
    temperature: float = 0.7
) -> pd.DataFrame:
    """
    Generate synthetic data based on the input DataFrame.
    
    Args:
        df: Input DataFrame with original data
        n_samples: Number of synthetic samples to generate
        temperature: LLM temperature parameter (0.0-1.0)
        
    Returns:
        DataFrame containing synthetic data
        
    Raises:
        ValueError: If n_samples is less than 1
    """
    ...
```

### JavaScript/React

- Use functional components with hooks
- Follow ESLint configuration
- Use meaningful variable and function names
- Prefer named exports over default exports

```javascript
// Good
export const DataExplorer = ({ onUpload, data }) => {
  const [isLoading, setIsLoading] = useState(false);
  
  const handleFileUpload = useCallback(async (file) => {
    setIsLoading(true);
    try {
      await onUpload(file);
    } finally {
      setIsLoading(false);
    }
  }, [onUpload]);
  
  return (/* ... */);
};
```

## Testing

### Backend Testing

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=. --cov-report=html

# Run specific test file
uv run pytest tests/test_generation.py
```

### Frontend Testing

```bash
cd frontend

# Run tests
npm test

# Run with coverage
npm test -- --coverage
```

## Documentation

- Update README.md for significant changes
- Add docstrings to new functions/classes
- Update API documentation for new endpoints
- Include inline comments for complex logic

## Questions?

If you have questions about contributing, please:

1. Check existing issues and discussions
2. Open a new issue with the `question` label
3. Be as specific as possible about your question

Thank you for contributing to MedGen! 🎉
