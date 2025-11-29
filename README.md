# Knowledge Base AI Agent 🤖

An AI agent that answers questions using **your company documents** (PDFs).  
Built for the 48-Hour AI Agent Development Challenge.

## Overview

Employees often struggle to find answers in long policy documents, handbooks, and SOPs.  
This agent lets you:

- Upload internal documents (PDF)
- Automatically index and embed them
- Ask natural language questions
- Get precise answers grounded in your documents

## Features

- PDF document upload
- Automatic text extraction and chunking
- Embedding-based similarity search
- Retrieval-Augmented Generation (RAG) using OpenAI GPT
- Shows retrieved context for transparency

## Tech Stack

- **UI:** Streamlit
- **AI Models:** OpenAI `gpt-4o-mini`, `text-embedding-3-small`
- **Language:** Python
- **Libraries:** `streamlit`, `openai`, `PyPDF2`, `numpy`

## How to Run Locally

```bash
git clone <your-repo-url>
cd knowledge-base-agent
pip install -r requirements.txt
```
