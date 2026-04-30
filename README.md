# 🏦 Autonomous Customer Support Triage (Agentic AI Workflow)

An enterprise-grade, multi-agent AI system built with **LangGraph** and **Google Gemini** that dynamically triages, resolves, and escalates real-world customer support tickets. 

Rather than building a standard, linear chatbot, this project implements a **Cognitive Routing State Machine**. It ingests incoming banking tickets, mathematically categorizes their intent, and makes autonomous routing decisions to either resolve routine issues instantly or escalate high-risk cases (like fraud) to human teams.

## 🚀 Business Value
* **Reduces MTTR (Mean Time to Resolution):** Automates Tier-1 support, instantly handling routine queries.
* **Mitigates Risk:** Instantly identifies and escalates urgent issues (e.g., compromised accounts) to human risk teams, bypassing standard queues.
* **Structured Data:** Forces LLM outputs into strict JSON schemas, ready for SQL database insertion without hallucinations.

## ✨ Key Engineering Features
* **Agentic Routing:** Utilizes LangGraph to build a cyclical workflow where an AI "Manager" evaluates data and conditionally routes it to specialized sub-agents.
* **Deterministic AI (Pydantic):** Prevents LLM hallucinations by forcing the categorization agent to select from a strict set of predefined banking intents.
* **Production-Ready Data Handling:** Bypasses dummy data by directly ingesting the industry-standard **PolyAI Banking77** dataset using Pandas.
* **API Rate Limit & Output Parsing:** Implements sleep timers to respect Google's Free-Tier API limits and uses `StrOutputParser` to strip invisible cryptographic metadata from the LLM's responses.

## 🛠️ Tech Stack
* **Orchestration:** LangGraph, LangChain
* **LLM Engine:** Google Gemini 1.5 Flash (`gemini-flash-latest`)
* **Data Validation:** Pydantic
* **Data Manipulation:** Pandas (PolyAI Banking77 Dataset)

## 🧠 System Architecture

1. **The State Memory:** A dictionary holding the ticket ID, raw customer message, intent category, and a boolean `requires_human` flag.
2. **The Triage Agent (The Brain):** Analyzes the ticket, extracts the banking intent, and evaluates urgency.
3. **The Router (Conditional Edge):** Checks the `requires_human` flag and directs traffic.
4. **The Auto-Responder Agent:** Drafts polite, context-aware resolutions for routine issues.
5. **The Escalation Agent:** Synthesizes 4-paragraph complaints into 1-sentence TL;DR summaries for human revie
