## 📌 Description

An **AI-powered multi-step procurement agent** that autonomously negotiates with suppliers via email or chat. It optimizes decisions using internal price policies, REST API integrations, and autonomous reasoning strategies. The agent handles multiple stages of procurement:

- **Search for suppliers**
- **Negotiate quotes**
- **Embed and compare offers**
- **Draft and send emails**
- **Handle exceptions**
- **Update internal databases**

This project aims to replicate real-world enterprise agent use cases and will be open-sourced for community contribution and feedback.

---

## 🚀 Features

- AI-powered negotiation and follow-up drafting
- REST API call orchestration and optimization
- Exception detection and escalation
- Supplier response embedding and similarity matching
- Quote storage and ranking in a vector DB
- Email and/or chat interaction
- Stateless + stateful memory (ReAct / LangChain memory)
- Cost-aware policy optimization

---

## 🧠 Tech Stack

| Component       | Technology                                      |
| --------------- | ----------------------------------------------- |
| Agent Framework | LangChain / AutoGPT                             |
| Prompt Logic    | ReAct, Memory Chains, Prompt Engineering        |
| Embeddings      | FAISS / Chroma (free)                           |
| Backend         | FastAPI / Flask + PostgreSQL                    |
| Email API       | Gmail API (OAuth2) / SMTP (Mailgun/SuperMailer) |
| Deployment      | Docker + Fly.io / Render.com (free tiers)       |
| Hosting DB      | Supabase (free PostgreSQL + API)                |

---

## 🧭 Strategy

1. **MVP First**: Email bot → quote embedder → quote comparator → DB writer.
2. **Agentic Loop**: Integrate LangChain agent to make decisions with tools.
3. **Scale Vector Search**: Use FAISS/Chroma locally or Supabase’s pgvector.
4. **Open-Source Friendly**: All tools must run locally or on free-tier services.
5. **Deployment Lite**: Dockerized + deployed via free-tier platforms like Fly.io or Render.
6. **Extensibility**: Modular functions, clear API design, config files for tools.

---

## 🛠️ Process Plan

### Phase 1: Bootstrap

- [ ] Set up GitHub repo with MIT license
- [ ] Scaffold project with FastAPI + Docker + FAISS
- [ ] Configure Gmail API or SMTP (sandbox email)
- [ ] Add REST API endpoints for testing (quotes, suppliers, etc.)

### Phase 2: Agent Framework Integration

- [ ] Integrate LangChain agent loop
- [ ] Build memory (ConversationBufferMemory or RedisMemory)
- [ ] Add tools for: REST API calls, vector DB search, policy check
- [ ] Define ReAct-style prompts

### Phase 3: Embedding & Comparison

- [ ] Use sentence-transformers to embed supplier quotes
- [ ] Store vectors using FAISS or ChromaDB
- [ ] Add similarity threshold matching logic

### Phase 4: Email Interaction

- [ ] Build email parsing layer (using `imaplib` or Gmail API)
- [ ] Draft response generator (LangChain LLMChain)
- [ ] Implement exception handling and escalation

### Phase 5: Quote Database

- [ ] Create a PostgreSQL DB schema for quotes and suppliers
- [ ] Integrate Supabase (or local pg via Docker)
- [ ] Build CRUD APIs for quote history

### Phase 6: Deployment & Docs

- [ ] Dockerize the app fully
- [ ] Deploy to Fly.io or Render (CI/CD optional)
- [ ] Create full README and open-source CONTRIBUTING.md
- [ ] Add unit tests for all major components

---

## ✅ Checklist

### Setup

- [X] GitHub Repo Initialized
- [X] Python project scaffolded (FastAPI + poetry/pipenv)
- [X] `.env` config file for secrets

### AI/Agent

- [ ] LangChain Agent Setup
- [ ] Tools created: EmailSenderTool, QuoteEmbedderTool, APICallTool
- [ ] Memory chain enabled

### Embedding & DB

- [ ] SentenceTransformer integration
- [ ] FAISS/Chroma vector DB
- [ ] PostgreSQL tables for quotes and suppliers

### Email Layer

- [ ] SMTP/Gmail API integration
- [ ] Parsing inbound emails
- [ ] Auto-reply with LLM-generated follow-up

### Deployment

- [ ] Dockerfile and docker-compose
- [ ] Deployment scripts (Fly.io or Render)
- [ ] Database hosted (Supabase or Dockerized pg)

---

## 📂 Project Structure (Suggested)

```bash
autonomous-procurement-agent/
├── app/
│   ├── agents/
│   ├── email/
│   ├── embeddings/
│   ├── api/
│   ├── db/
│   └── utils/
├── tests/
├── requirements.txt / pyproject.toml
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## 💡 Future Ideas

- Slack / Teams bot integration
- Frontend dashboard (React + Tailwind)
- Custom vector-based supplier ranking
- LLM fine-tuning with procurement datasets
- LLM cost tracking and logging

---

## 🤝 Contributing

This project will be open-source. Contributions welcome!

- Fork the repo
- Create a feature branch
- Submit a PR
- Write tests if possible

---

## 🧳 Resources

- LangChain Docs
- [FAISS](https://github.com/facebookresearch/faiss)
- Gmail API Quickstart
- [Supabase](https://supabase.com/)
- [Fly.io (Free Deployment)](https://fly.io/)
- [ChromaDB](https://www.trychroma.com/)
