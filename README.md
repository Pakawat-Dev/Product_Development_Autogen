# 🚀 Product Development Agent

A Streamlit web application that uses AI agents to generate comprehensive MVP briefs through collaborative multi-agent workflows.

## Features

- **Multi-Agent System**: 5 specialized AI agents working together
- **Token Usage Tracking**: Real-time monitoring of input/output tokens
- **Interactive UI**: Clean Streamlit interface with relevant icons
- **Multiple LLM Support**: OpenAI and Anthropic models

## Agents

- 🔍 **UX Researcher**: Analyzes user needs and pain points
- 🏗️ **Tech Lead**: Designs architecture and technical solutions
- ✅ **QA Compliance**: Reviews safety, privacy, and regulatory requirements
- 📋 **Product Manager**: Orchestrates and coordinates the workflow
- ✍️ **Tech Writer**: Synthesizes outputs into final deliverable

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create `.env` file with API keys:
```
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

3. Run the application:
```bash
streamlit run streamlit_app.py
```

## Usage

1. Enter your product brief request in the sidebar
2. Click "🎯 Generate Brief" 
3. View token usage metrics and final MVP brief
4. Monitor agent status in real-time

## Requirements

- Python 3.8+
- OpenAI API key
- Anthropic API key
- Internet connection
