# Security Policy

## Supported Versions
We support the latest version of the main branch for security fixes.

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability
We take the security of **RAG Document Assistant** seriously.

If you discover a potential security vulnerability (such as an API key leakage risk, injection vulnerability, or unsafe deserialization):

1. **Do NOT open a public issue.**
2. Please privately disclose details by contacting the project maintainers or creating a Private Vulnerability Report on GitHub.
3. Provide a detailed summary, including:
   - Type of issue (e.g. prompt injection, unsafe input handling).
   - Steps to reproduce.
   - Proof of concept or potential impact.

We will review the issue and release a patch as quickly as possible.

## Best Practices for Users
- **Never commit your `.env` file**: `.env` is listed in `.gitignore` by default. Always keep API keys secret.
- **Run in virtual environments**: Avoid installing packages globally with administrative privileges.
- **Use trusted documents**: Be mindful of untrusted third-party documents containing adversarial instructions (prompt injections).
