# 🤖 AI Document Summarizer

Summarize and chat with your documents using multiple AI providers (Groq, OpenAI, Google Gemini, Anthropic Claude).

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
[![Deploy to Streamlit](https://img.shields.io/badge/Streamlit-Deploy-FF4B4B?logo=streamlit)](https://share.streamlit.io)

## 🔗 Live Demo

**[👉 Try it now on Streamlit Cloud](https://ai-document-summarizer-64dsrqnvjzreb4qutyjxrc.streamlit.app/)**

> ⚠️ You need to configure API keys in the Streamlit Cloud secrets settings.

## ✨ Features

- **📄 Multi-format Support**: Upload PDF or TXT files
- **💬 Document Q&A**: Chat with your documents in Vietnamese
- **🔄 Multiple AI Providers**: Choose from Groq, OpenAI, Google Gemini, Anthropic Claude
- **📝 Smart Summarization**: Get structured summaries with key points and bullet points
- **🎨 Modern UI**: AI-style dark theme with glowing gradients

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/NBA-Lion/ai-document-summarizer.git
cd ai-document-summarizer
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure API Keys

Create a `.env` file in the project root:
```env
# Groq (Free, fast)
GROQ_API_KEY=your_groq_api_key

# OpenAI (Optional)
OPENAI_API_KEY=your_openai_api_key

# Google Gemini (Optional)
GEMINI_API_KEY=your_gemini_api_key

# Anthropic Claude (Optional)
ANTHROPIC_API_KEY=your_anthropic_api_key
```

### 4. Run the app
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

## 📖 Usage

1. **Select AI Provider**: Choose your preferred AI provider from the sidebar
2. **Select Model**: Pick a model based on your provider
3. **Upload Document**: Upload a PDF or TXT file (max 10MB)
4. **Get Summary**: Click "📝 Tóm tắt tài liệu" to generate summary
5. **Chat with Document**: Ask questions about your document in the chat section

## 🤖 Supported AI Models

| Provider | Models |
|----------|--------|
| **Groq** | qwen3.8-27b, qwen3.6-27b, gpt-oss-120b |
| **OpenAI** | gpt-4o, gpt-4-turbo, gpt-3.5-turbo |
| **Google** | gemini-1.5-pro, gemini-1.5-flash |
| **Anthropic** | claude-3.5-sonnet, claude-3-opus |

## 🛠️ API Key Setup

### Groq (Recommended - Free)
1. Go to [console.groq.com](https://console.groq.com)
2. Create an account or sign in
3. Navigate to API Keys and create a new key
4. Free tier includes 30 requests/minute

### OpenAI
1. Go to [platform.openai.com](https://platform.openai.com)
2. Create an account and add credits
3. Generate an API key

### Google Gemini
1. Go to [aistudio.google.com](https://aistudio.google.com)
2. Get your API key from AI Studio
3. Free tier available with limits

### Anthropic Claude
1. Go to [console.anthropic.com](https://console.anthropic.com)
2. Create an account and add credits
3. Generate an API key

## 📁 Project Structure

```
ai-document-summarizer/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── .env.example       # Example environment variables
├── .gitignore         # Git ignore rules
└── README.md          # This file
```

## ⚙️ Configuration

### Chunk Size
The app automatically splits large documents into chunks for processing:
- **Small documents** (< 7000 tokens): Processed in one go
- **Large documents**: Split into ~5000 token chunks

### Model Selection
- **Groq**: Best for free users, fast inference
- **OpenAI GPT-4o**: Best for quality
- **Gemini 1.5 Flash**: Good balance of speed and quality
- **Claude 3.5 Sonnet**: Great for complex analysis

## 📝 License

This project is licensed under the MIT License.

##  Acknowledgments

Built with Streamlit, Groq, OpenAI, Google Gemini, and Anthropic.
