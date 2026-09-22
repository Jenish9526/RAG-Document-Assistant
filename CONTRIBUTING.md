# Contributing to RAG Document Assistant

Thank you for your interest in contributing to **RAG Document Assistant**! 🎉

We welcome contributions of all kinds: bug reports, documentation updates, feature requests, and pull requests.

---

## Code of Conduct
By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

---

## How Can I Contribute?

### 1. Reporting Bugs
- Search existing [Issues](https://github.com/) to see if the problem has already been reported.
- If not, open a new issue using our **Bug Report** template.
- Include complete details: operating system, Python version, steps to reproduce, and relevant logs.

### 2. Suggesting Enhancements
- Check if your idea is already being discussed in existing issues.
- Open a new issue using our **Feature Request** template.

### 3. Submitting Pull Requests
Follow this workflow to submit code contributions:

1. **Fork the repository** on GitHub.
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/<your-username>/RAG_Document_Assistant.git
   cd RAG_Document_Assistant
   ```
3. **Create a virtual environment**:
   ```bash
   python -m venv venv
   # Windows:
   .\venv\Scripts\Activate.ps1
   # macOS/Linux:
   source venv/bin/activate
   ```
4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
5. **Create a descriptive topic branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```
6. **Make your changes** cleanly:
   - Follow PEP 8 guidelines.
   - Preserve comments and architectural isolation (e.g. keeping LLM calls isolated in `llm_service.py`).
7. **Run the test suite**:
   ```bash
   python -m unittest discover tests
   ```
   Ensure all tests pass before committing.
8. **Commit and Push**:
   ```bash
   git add .
   git commit -m "feat: add support for new feature"
   git push origin feature/your-feature-name
   ```
9. **Open a Pull Request** against `main` on the original repository.

---

## Code Style & Standards
- Keep functions modular with clear type hints.
- Do not commit secrets, `.env` files, or binary vector database artifacts (`index.faiss`, `metadata.pkl`).
- Add unit tests for any new features or bug fixes in `tests/`.

---

Thank you for helping make RAG Document Assistant better!
