# Contributing to Autonomous Procurement Agent

Thank you for your interest in contributing to this project!

---

## 🧱 How to Contribute

### 1. Fork the Repository

Click the **Fork** button on the top right of the GitHub page to create your own copy of the project.

### 2. Clone Your Fork

```bash
git clone https://github.com/kashewknutt/autonomous-procurement-agent.git
cd autonomous-procurement-agent
```
### 3. Create a Feature Branch

```bash
git checkout -b feat/your-feature-name
```
### 4. Install Dependencies & Set Up Locally

Follow the instructions in the README to set up the virtual environment and local PostgreSQL database.

### 5. Make Your Changes

Stick to the existing folder structure:

- `app/agents/` for agent logic
    
- `app/api/` for FastAPI routes
    
- `app/db/` for DB models and operations
    
- `app/embeddings/` for vector logic

### 7. Commit with Conventional Commit Messages

Example format:

```bash
git commit -m "feat(agent): added new tool for quote summarization"
```

### 8. Push and Open a Pull Request

```bash
git push origin feat/your-feature-name
```

Then open a PR from your fork → main repo. Fill out the PR template (if available), describe what you did, and why.

---

## 🧼 Code Style

- Python 3.10+
    
- Stick to **black** formatting
    
- Type hints wherever possible
    
- Keep functions small and modular

---

## 💬 Questions or Suggestions?

Open an [Issue](https://github.com/kashewknutt/autonomous-procurement-agent/issues) or join the discussion via PR comments.

We appreciate all contributions! 🙌