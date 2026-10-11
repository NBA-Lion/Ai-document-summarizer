"""AI Document Summarizer & Q&A (Streamlit + Multiple AI Providers).

A single-page Streamlit app that lets the user upload a PDF or TXT document,
get a structured Vietnamese summary, and chat with the document via various AI providers.
"""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from groq import (
    APIConnectionError,
    APIError,
    AuthenticationError,
    Groq,
    RateLimitError,
)
from pypdf import PdfReader

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import google.generativeai as genai
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False

# Load .env if present.
load_dotenv()

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AI Document Summarizer",
    page_icon="🤖",
    layout="wide",
)

# Custom AI-style CSS Theme
st.markdown("""
<style>
    /* Gradient background cho entire app */
    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    }
    
    /* Main content background */
    .main .block-container {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #0f0f1a 100%);
        border-right: 1px solid rgba(102, 126, 234, 0.3);
    }
    
    /* Glowing gradient buttons */
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border: none;
        border-radius: 25px;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
        font-weight: 600;
        color: white !important;
    }
    
    .stButton > button:hover {
        box-shadow: 0 6px 25px rgba(102, 126, 234, 0.6);
        transform: translateY(-2px);
    }
    
    /* Primary button special glow */
    .stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #f093fb 0%, #f5576c 100%);
        box-shadow: 0 4px 20px rgba(245, 87, 108, 0.4);
    }
    
    .stButton > button[kind="primary"]:hover {
        box-shadow: 0 6px 30px rgba(245, 87, 108, 0.6);
    }
    
    /* Chat message styling */
    [data-testid="stChatMessage"] {
        border-radius: 15px;
        padding: 15px;
        margin: 10px 0;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* User message - right aligned gradient */
    [data-testid="stChatMessage"][data-testid="stChatMessage-user"] {
        background: linear-gradient(135deg, #667eea22 0%, #764ba222 100%);
        border-left: 3px solid #667eea;
    }
    
    /* Assistant message - left aligned */
    [data-testid="stChatMessage"][data-testid="stChatMessage-assistant"] {
        background: linear-gradient(135deg, #f093fb11 0%, #f5576c11 100%);
        border-left: 3px solid #f5576c;
    }
    
    /* Tabs styling */
    .stTabs [data-testid="stTab"] {
        background: transparent;
        border-radius: 10px 10px 0 0;
        padding: 10px 25px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-testid="stTab"]:hover {
        background: rgba(102, 126, 234, 0.2);
    }
    
    .stTabs [data-testid="stTab"][aria-selected="true"] {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Metrics cards */
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 15px;
        border: 1px solid rgba(102, 126, 234, 0.2);
    }
    
    [data-testid="stMetricLabel"] {
        color: #a0a0b0 !important;
    }
    
    [data-testid="stMetricValue"] {
        color: #667eea !important;
        font-weight: 700;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        border: 2px dashed rgba(102, 126, 234, 0.5);
        border-radius: 15px;
        padding: 20px;
        background: rgba(102, 126, 234, 0.05);
    }
    
    /* Success/Info/Warning boxes */
    .stAlert {
        border-radius: 12px;
        border: none;
    }
    
    /* Title styling */
    h1 {
        background: linear-gradient(90deg, #667eea 0%, #f093fb 50%, #f5576c 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 800;
    }
    
    /* Subheader styling */
    h2, h3 {
        color: #e0e0e0 !important;
    }
    
    /* Spinner custom */
    [data-testid="stSpinner"] {
        color: #667eea;
    }
    
    /* Selectbox/Dropdown */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        border: 1px solid rgba(102, 126, 234, 0.3);
    }
    
    /* Text input */
    .stTextInput > div > div {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        border: 1px solid rgba(102, 126, 234, 0.3);
    }
    
    /* Chat input */
    [data-testid="stChatInput"] {
        background: rgba(255, 255, 255, 0.08);
        border-radius: 25px;
        border: 1px solid rgba(102, 126, 234, 0.4);
    }
    
    /* Divider */
    hr {
        border-color: rgba(102, 126, 234, 0.2);
    }
    
    /* Caption/Footer */
    .stCaption {
        color: #6c6c7c !important;
    }
    
    /* Code blocks */
    code {
        background: rgba(102, 126, 234, 0.15) !important;
        border-radius: 5px;
        padding: 2px 6px;
        color: #f093fb !important;
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.05);
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(102, 126, 234, 0.5);
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_CHARS: int = 12_000  # safe truncation budget for the LLM context
SCAN_THRESHOLD: int = 20  # characters below which a PDF is treated as scanned

# AI Providers configuration
PROVIDERS: dict[str, dict] = {
    "groq": {
        "name": "Groq",
        "icon": "⚡",
        "api_key_env": "GROQ_API_KEY",
        "models": [
            "qwen/qwen3.8-27b",
            "openai/gpt-oss-120b",
            "qwen/qwen3.6-27b",
        ],
        "model_display": {
            "qwen/qwen3.8-27b": "Qwen 3.8 - 27B",
            "openai/gpt-oss-120b": "GPT OSS - 120B",
            "qwen/qwen3.6-27b": "Qwen 3.6 - 27B",
        },
        "default_model": "qwen/qwen3.8-27b",
    },
    "openai": {
        "name": "OpenAI",
        "icon": "🤖",
        "api_key_env": "OPENAI_API_KEY",
        "models": [
            "gpt-4o",
            "gpt-4-turbo",
            "gpt-3.5-turbo",
        ],
        "model_display": {
            "gpt-4o": "GPT-4o",
            "gpt-4-turbo": "GPT-4 Turbo",
            "gpt-3.5-turbo": "GPT-3.5 Turbo",
        },
        "default_model": "gpt-4o",
    },
    "google": {
        "name": "Google",
        "icon": "💎",
        "api_key_env": "GEMINI_API_KEY",
        "models": [
            "gemini-1.5-pro",
            "gemini-1.5-flash",
        ],
        "model_display": {
            "gemini-1.5-pro": "Gemini 1.5 Pro",
            "gemini-1.5-flash": "Gemini 1.5 Flash",
        },
        "default_model": "gemini-1.5-flash",
    },
    "anthropic": {
        "name": "Anthropic",
        "icon": "🧠",
        "api_key_env": "ANTHROPIC_API_KEY",
        "models": [
            "claude-3-5-sonnet-20241022",
            "claude-3-opus-20240229",
        ],
        "model_display": {
            "claude-3-5-sonnet-20241022": "Claude 3.5 Sonnet",
            "claude-3-opus-20240229": "Claude 3 Opus",
        },
        "default_model": "claude-3-5-sonnet-20241022",
    },
}

PROVIDER_ORDER = ["groq", "openai", "google", "anthropic"]

SYSTEM_PROMPT_QA = (
    "Bạn là một trợ lý AI chuyên trả lời câu hỏi về một tài liệu được cung cấp. "
    "Hãy trả lời TRUNG THỰC, NGẮN GỌN (tối đa 200 từ) dựa trên nội dung tài liệu. "
    "Nếu không biết hoặc thông tin không có trong tài liệu, hãy nói rõ là không biết. "
    "Không bịa đặt thông tin."
)

DOCUMENT_BLOCK_TEMPLATE = (
    "Dưới đây là nội dung tài liệu mà người dùng đã tải lên. "
    "Hãy dùng nó làm ngữ cảnh chính để trả lời các câu hỏi tiếp theo.\n\n"
    "[TÀI LIỆU]\n{document}\n[/TÀI LIỆU]"
)


def _build_context_from_chunks(chunks: list[dict], max_chars: int = MAX_CHARS) -> str:
    """
    Build context string from chunks, respecting max character limit.
    Takes from beginning and distributes across chunks.
    """
    if not chunks:
        return ""
    
    if len(chunks) == 1:
        return chunks[0]['text'][:max_chars]
    
    # Distribute characters across first N chunks
    per_chunk = max_chars // min(3, len(chunks))
    context_parts = []
    total_chars = 0
    
    for i, chunk in enumerate(chunks[:3]):  # Max 3 chunks for initial context
        chunk_text = chunk['text']
        remaining = max_chars - total_chars
        if remaining <= 0:
            break
        
        # Take proportional amount
        take = min(per_chunk, remaining, len(chunk_text))
        context_parts.append(chunk_text[:take])
        total_chars += take
    
    return '\n\n'.join(context_parts)


# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------

def init_session_state() -> None:
    """Initialize st.session_state keys on first run."""
    defaults: dict[str, object] = {
        "messages": [],              # chat history (role/content dicts)
        "document_text": "",         # extracted + original document text
        "document_chunks": [],       # smart chunks for long documents
        "file_name": "",              # name of the currently loaded file
        "summary": "",               # last generated summary
        "provider": "groq",           # selected AI provider
        "model": "qwen/qwen3.8-27b", # selected model (per provider)
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def get_api_key(provider: str) -> str | None:
    """Return the API key for a specific provider from sidebar input, st.secrets, or env."""
    provider_config = PROVIDERS.get(provider, {})
    env_var = provider_config.get("api_key_env", f"{provider.upper()}_API_KEY")
    
    # Get from provider-specific session state
    session_key = f"api_key_{provider}"
    sidebar_key = (st.session_state.get(session_key, "") or "").strip()
    if sidebar_key:
        return sidebar_key
    
    # Try st.secrets
    try:
        secrets_key = st.secrets.get(env_var, "")  # type: ignore[attr-defined]
        if secrets_key:
            return str(secrets_key)
    except Exception:
        pass
    
    # Try environment variable
    env_key = os.getenv(env_var, "").strip()
    return env_key or None


def read_pdf(file) -> str:
    """Extract text from a PDF file-like object."""
    reader = PdfReader(file)
    pages: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            pages.append(text)
    return "\n\n".join(pages)


def read_txt(file) -> str:
    """Read a TXT file-like object as UTF-8 (ignoring decode errors)."""
    raw = file.read()
    if isinstance(raw, str):
        return raw
    return raw.decode("utf-8", errors="ignore")


def truncate_text(text: str, limit: int = MAX_CHARS) -> str:
    """Truncate text to `limit` characters with a marker if cut."""
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "\n\n[...Nội dung đã được cắt bớt để vừa token limit...]"


def is_pdf_scanned(extracted_text: str) -> bool:
    """Heuristic: a PDF with almost no extractable text is likely scanned."""
    return len(extracted_text.strip()) < SCAN_THRESHOLD


# ---------------------------------------------------------------------------
# Smart Chunking Functions
# ---------------------------------------------------------------------------

CHUNK_SIZE = 5_000   # Reduced to avoid Groq rate limit
CHUNK_OVERLAP = 300  # Smaller overlap for smaller chunks

def split_into_chunks(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """
    Split text into overlapping chunks with smart paragraph awareness.
    
    Returns list of dicts with 'text', 'index', and 'total' keys.
    """
    if not text.strip():
        return []
    
    # Clean and normalize text
    text = text.strip().replace('\r\n', '\n').replace('\r', '\n')
    
    # Split by paragraphs first to preserve context
    paragraphs = text.split('\n\n')
    
    chunks = []
    current_chunk = []
    current_size = 0
    chunk_index = 0
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
            
        para_len = len(para)
        
        # If single paragraph exceeds chunk size, split by sentences
        if para_len > chunk_size:
            # Save current chunk if not empty
            if current_chunk:
                chunks.append({
                    'text': '\n\n'.join(current_chunk),
                    'index': chunk_index,
                    'total': 0  # Will be updated later
                })
                chunk_index += 1
                current_chunk = []
                current_size = 0
            
            # Split long paragraph by sentences
            sentences = para.split('. ')
            temp_chunk = []
            temp_size = 0
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                    
                # Add period back (except for last sentence)
                sentence_with_period = sentence + ('.' if not sentence.endswith('.') else '')
                
                if temp_size + len(sentence_with_period) > chunk_size and temp_chunk:
                    # Save this sub-chunk
                    chunks.append({
                        'text': ' '.join(temp_chunk),
                        'index': chunk_index,
                        'total': 0
                    })
                    chunk_index += 1
                    temp_chunk = []
                    temp_size = 0
                
                temp_chunk.append(sentence_with_period)
                temp_size += len(sentence_with_period) + 1
            
            # Save remaining sentences
            if temp_chunk:
                chunks.append({
                    'text': ' '.join(temp_chunk),
                    'index': chunk_index,
                    'total': 0
                })
                chunk_index += 1
            
        elif current_size + para_len + 2 > chunk_size:
            # Current chunk is full, save it
            chunks.append({
                'text': '\n\n'.join(current_chunk),
                'index': chunk_index,
                'total': 0
            })
            chunk_index += 1
            
            # Start new chunk with overlap
            if overlap > 0 and current_chunk and current_chunk[-1]:
                # Include last paragraph as overlap
                current_chunk = [current_chunk[-1]]
                current_size = len(current_chunk[-1])
            else:
                current_chunk = []
                current_size = 0
            
            current_chunk.append(para)
            current_size += para_len + 2
            
        else:
            current_chunk.append(para)
            current_size += para_len + 2
    
    # Don't forget the last chunk
    if current_chunk:
        chunks.append({
            'text': '\n\n'.join(current_chunk),
            'index': chunk_index,
            'total': 0
        })
    
    # Update total count for all chunks
    total_chunks = len(chunks)
    for chunk in chunks:
        chunk['total'] = total_chunks
    
    return chunks


def find_relevant_chunks(chunks: list[dict], query: str, top_k: int = 3) -> list[dict]:
    """
    Find chunks most relevant to the query using keyword matching.
    Falls back to first chunks if no strong matches.
    """
    if not chunks:
        return []
    
    # Simple keyword-based relevance scoring
    query_words = set(query.lower().split())
    query_keywords = {
        'tổng quan', 'tóm tắt', 'giới thiệu', 'mục tiêu', 'kết luận',
        'phương pháp', 'kết quả', 'thảo luận', 'giới thiệu', 'bối cảnh',
        'định nghĩa', 'khái niệm', 'ví dụ', 'cách', 'làm sao', 'tại sao',
        'như thế nào', 'bao nhiêu', 'khi nào', 'ở đâu', 'ai'
    }
    
    scores = []
    for chunk in chunks:
        text_lower = chunk['text'].lower()
        words = set(text_lower.split())
        
        # Count matching keywords
        matches = len(query_words & words)
        
        # Boost for title-like patterns
        if any(marker in text_lower[:200] for marker in ['#', '##', '**', 'tiêu đề', 'chương', 'phần']):
            matches += 0.5
        
        scores.append((chunk, matches))
    
    # Sort by score descending
    scores.sort(key=lambda x: x[1], reverse=True)
    
    # Get top_k chunks, or at least the first 2 chunks if no strong matches
    if scores and scores[0][1] > 0:
        relevant = [s[0] for s in scores[:top_k]]
    else:
        # No keyword matches, take first chunks
        relevant = chunks[:min(top_k, len(chunks))]
    
    # Sort back by original order
    relevant.sort(key=lambda x: x['index'])
    
    return relevant


def build_summary_prompt(document: str) -> str:
    """Build the user prompt used for the summary action."""
    return (
        "Hãy đọc tài liệu sau đây và tạo một bản tóm tắt bằng tiếng Việt "
        "(hoặc theo ngôn ngữ của tài liệu nếu không phải tiếng Việt) "
        "với đúng ba phần, sử dụng định dạng Markdown:\n\n"
        "## Tổng quan\n"
        "(Mô tả ngắn gọn nội dung tổng thể của tài liệu.)\n\n"
        "## Các điểm chính\n"
        "(Liệt kê các điểm chính dưới dạng danh sách gạch đầu dòng.)\n\n"
        "## Kết luận\n"
        "(Tóm tắt kết luận hoặc thông điệp cốt lõi của tài liệu.)\n\n"
        "--- TÀI LIỆU ---\n"
        f"{document}\n"
        "--- HẾT TÀI LIỆU ---"
    )


def handle_api_error(error: Exception, provider: str) -> str:
    """Translate API exceptions into localized, user-friendly messages."""
    provider_name = PROVIDERS.get(provider, {}).get("name", provider.title())
    
    # Groq errors
    if isinstance(error, AuthenticationError):
        return f"❌ API key không hợp lệ hoặc đã hết hạn. Vui lòng kiểm tra lại {provider_name} API key."
    if isinstance(error, RateLimitError):
        return f"⚠️ Đã vượt quá giới hạn tốc độ (rate limit). Vui lòng đợi một chút rồi thử lại."
    if isinstance(error, APIConnectionError):
        return f"🔌 Không thể kết nối tới {provider_name}. Vui lòng kiểm tra kết nối mạng."
    if isinstance(error, APIError):
        return f"❌ Lỗi từ {provider_name} API: {error}"
    
    # OpenAI errors
    if "authentication" in str(error).lower() or "api_key" in str(error).lower():
        return f"❌ API key không hợp lệ hoặc đã hết hạn. Vui lòng kiểm tra lại {provider_name} API key."
    
    # Anthropic errors
    if hasattr(error, "status_code"):
        if error.status_code == 401:
            return f"❌ API key không hợp lệ hoặc đã hết hạn. Vui lòng kiểm tra lại {provider_name} API key."
        if error.status_code == 429:
            return f"⚠️ Đã vượt quá giới hạn tốc độ (rate limit). Vui lòng đợi một chút rồi thử lại."
    
    return f"❌ Đã xảy ra lỗi: {error}"


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## ⚙️ Cấu hình")

    # Provider selection
    provider_options = [p for p in PROVIDER_ORDER]
    
    current_provider_index = provider_options.index(st.session_state.provider) if st.session_state.provider in provider_options else 0
    selected_provider = st.selectbox(
        "🔌 AI Provider",
        options=provider_options,
        format_func=lambda x: f"{PROVIDERS[x]['icon']} {PROVIDERS[x]['name']}",
        index=current_provider_index,
        key="provider_select",
    )
    st.session_state.provider = selected_provider

    provider_config = PROVIDERS[selected_provider]
    
    # Provider-specific API key input
    api_key_label = f"{provider_config['icon']} {provider_config['name']} API Key"
    help_texts = {
        "groq": "Lấy key miễn phí tại console.groq.com/keys",
        "openai": "Lấy key tại platform.openai.com/api-keys",
        "google": "Lấy key tại aistudio.google.com/app/apikey",
        "anthropic": "Lấy key tại console.anthropic.com/settings/keys",
    }
    
    session_key = f"api_key_{selected_provider}"
    current_api_key = st.session_state.get(session_key, "") or os.getenv(provider_config["api_key_env"], "")
    
    api_key_input = st.text_input(
        api_key_label,
        type="password",
        value=current_api_key,
        help=help_texts.get(selected_provider, "Nhập API key"),
        key=f"api_key_input_{selected_provider}",
    )
    st.session_state[session_key] = api_key_input

    # Model selection for the current provider
    provider_models = provider_config["models"]
    
    def format_model(model_id: str) -> str:
        return provider_config["model_display"].get(model_id, model_id)
    
    # Get current model, default to provider's default if not set
    current_model = st.session_state.get("model", "")
    if current_model not in provider_models:
        current_model = provider_config["default_model"]
    
    selected_model = st.selectbox(
        "🧠 Model",
        options=provider_models,
        format_func=format_model,
        index=provider_models.index(current_model) if current_model in provider_models else 0,
        key="model_select",
    )
    st.session_state.model = selected_model

    st.markdown("---")

    if st.button("🗑️ Xóa lịch sử chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.summary = ""
        st.session_state.file_name = ""
        st.session_state.document_text = ""
        st.session_state.document_chunks = []
        # Reset the file uploader by bumping its key.
        st.session_state["file_uploader_reset"] = (
            st.session_state.get("file_uploader_reset", 0) + 1
        )
        st.rerun()

    st.markdown("---")
    st.markdown(
        "💡 **Mẹo:** Bạn có thể đặt API key trong file `.env` "
        "thay vì nhập trực tiếp ở đây."
    )


# ---------------------------------------------------------------------------
# Top-of-page guard: API key must be present before continuing
# ---------------------------------------------------------------------------

provider = st.session_state.provider
api_key = get_api_key(provider)
provider_name = PROVIDERS[provider]["name"]

if not api_key:
    env_var = PROVIDERS[provider]["api_key_env"]
    st.info(
        f"👋 Vui lòng nhập **{provider_name} API Key** ở thanh bên trái "
        f"hoặc tạo file `.env` với biến `{env_var}`."
    )
    st.stop()

# Initialize the appropriate client based on provider
if provider == "groq":
    client = Groq(api_key=api_key)
elif provider == "openai":
    client = OpenAI(api_key=api_key)
elif provider == "anthropic":
    anthropic_client = anthropic.Anthropic(api_key=api_key)
elif provider == "google":
    genai.configure(api_key=api_key)
    google_client = genai


def create_chat_completion(prompt_or_messages, model: str, temperature: float = 0.3, stream: bool = False, system_prompt: str = None, max_tokens: int = 2048):
    """
    Unified function to create chat completions across all providers.
    
    Args:
        prompt_or_messages: Either a string prompt (for simple calls) or list of messages
        model: The model ID
        temperature: Temperature setting
        stream: Whether to stream the response
        system_prompt: Optional system prompt to prepend
        max_tokens: Maximum tokens in response (default 2048)
    
    Returns:
        Response object (type varies by provider)
    """
    current_provider = st.session_state.provider
    
    if current_provider == "groq":
        if isinstance(prompt_or_messages, str):
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt_or_messages})
        else:
            messages = prompt_or_messages
        
        return client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            stream=stream,
            max_tokens=max_tokens,
        )
    
    elif current_provider == "openai":
        if isinstance(prompt_or_messages, str):
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt_or_messages})
        else:
            messages = prompt_or_messages
        
        return client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            stream=stream,
            max_tokens=max_tokens,
        )
    
    elif current_provider == "anthropic":
        # Convert messages format for Anthropic
        if isinstance(prompt_or_messages, str):
            messages = []
            if system_prompt:
                messages.append({"role": "user", "content": system_prompt + "\n\n" + prompt_or_messages})
            else:
                messages.append({"role": "user", "content": prompt_or_messages})
        else:
            # Convert OpenAI-style messages to Anthropic format
            anthropic_messages = []
            for msg in prompt_or_messages:
                if msg["role"] == "system":
                    # Prepend system to first user message
                    continue
                anthropic_messages.append({
                    "role": "user" if msg["role"] == "user" else "assistant",
                    "content": msg["content"]
                })
            messages = anthropic_messages
        
        extra_kwargs = {}
        if stream:
            extra_kwargs["stream"] = True
        
        return anthropic_client.messages.create(
            model=model,
            max_tokens=4096,
            messages=messages,
            temperature=temperature,
            **extra_kwargs
        )
    
    elif current_provider == "google":
        # Convert messages format for Google Gemini
        if isinstance(prompt_or_messages, str):
            full_prompt = prompt_or_messages
            if system_prompt:
                full_prompt = system_prompt + "\n\n" + prompt_or_messages
            model_gemini = google_client.generate_model(model)
            return model_gemini.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    stream=stream,
                )
            )
        else:
            # Convert conversation history to a single prompt for Gemini
            combined_content = []
            for msg in prompt_or_messages:
                if msg["role"] == "system":
                    combined_content.insert(0, msg["content"])
                else:
                    role = "User" if msg["role"] == "user" else "Assistant"
                    combined_content.append(f"{role}: {msg['content']}")
            combined_content.append("Assistant: ")
            full_prompt = "\n\n".join(combined_content)
            
            model_gemini = google_client.generate_model(model)
            return model_gemini.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    stream=stream,
                )
            )


# ---------------------------------------------------------------------------
# Main area: title, file uploader
# ---------------------------------------------------------------------------

st.title("📄 AI Document Summarizer & Q&A")
st.markdown(
    "Tải lên tài liệu PDF hoặc TXT để nhận bản **tóm tắt nhanh** và **hỏi đáp** "
    "trực tiếp với nội dung."
)

# Model badge & stats row
col_badge, col_docs, col_msgs, col_chars = st.columns([2, 3, 2, 2])

with col_badge:
    provider_config = PROVIDERS[st.session_state.provider]
    provider_icon = provider_config["icon"]
    provider_name = provider_config["name"]
    model_display = provider_config["model_display"].get(st.session_state.model, st.session_state.model)
    
    st.markdown(f"""
    <div style="
        background: linear-gradient(90deg, #667eea22 0%, #764ba222 100%);
        border: 1px solid #667eea44;
        border-radius: 20px;
        padding: 8px 16px;
        text-align: center;
        margin-bottom: 10px;
    ">
        <span style="font-size: 12px; color: #888;">{provider_icon} {provider_name}</span><br>
        <span style="font-size: 14px; font-weight: 700; color: #667eea;">{model_display}</span>
    </div>
    """, unsafe_allow_html=True)

with col_docs:
    doc_name = st.session_state.file_name or "Chưa có tài liệu"
    doc_color = "#4ade80" if st.session_state.file_name else "#888"
    st.markdown(f"""
    <div style="
        background: rgba(255, 255, 255, 0.03);
        border-radius: 12px;
        padding: 10px 15px;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.05);
    ">
        <span style="font-size: 12px; color: #888;">📄 Tài liệu</span><br>
        <span style="font-size: 13px; font-weight: 600; color: {doc_color};">{doc_name[:30]}{'...' if len(doc_name) > 30 else ''}</span>
    </div>
    """, unsafe_allow_html=True)

with col_msgs:
    msg_count = len([m for m in st.session_state.messages if m.get("role") != "system"])
    st.markdown(f"""
    <div style="
        background: rgba(255, 255, 255, 0.03);
        border-radius: 12px;
        padding: 10px 15px;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.05);
    ">
        <span style="font-size: 12px; color: #888;">💬 Câu hỏi</span><br>
        <span style="font-size: 18px; font-weight: 700; color: #f093fb;">{msg_count}</span>
    </div>
    """, unsafe_allow_html=True)

with col_chars:
    char_count = len(st.session_state.document_text)
    char_display = f"{char_count:,}" if char_count else "0"
    st.markdown(f"""
    <div style="
        background: rgba(255, 255, 255, 0.03);
        border-radius: 12px;
        padding: 10px 15px;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.05);
    ">
        <span style="font-size: 12px; color: #888;">📊 Ký tự</span><br>
        <span style="font-size: 14px; font-weight: 700; color: #f5576c;">{char_display}</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

uploader_key = f"file_uploader_{st.session_state.get('file_uploader_reset', 0)}"
uploaded = st.file_uploader(
    "📤 Tải lên tài liệu",
    type=["pdf", "txt"],
    key=uploader_key,
)

if uploaded is not None:
    if uploaded.name != st.session_state.file_name:
        try:
            name_lower = uploaded.name.lower()
            if uploaded.type == "application/pdf" or name_lower.endswith(".pdf"):
                extracted = read_pdf(uploaded)
                if is_pdf_scanned(extracted):
                    st.error(
                        "❌ PDF dạng ảnh quét, không trích xuất được văn bản. "
                        "Vui lòng dùng PDF có lớp text hoặc chạy OCR trước."
                    )
                    st.session_state.document_text = ""
                    st.session_state.file_name = ""
                else:
                    # Store full text (no truncation for chunking)
                    st.session_state.document_text = extracted
                    # Create smart chunks
                    chunks = split_into_chunks(extracted)
                    st.session_state.document_chunks = chunks
                    st.session_state.file_name = uploaded.name
                    st.session_state.summary = ""
                    st.session_state.messages = []
                    # Pre-seed chat history with a system message (hidden from UI).
                    # Use first chunk(s) for initial context
                    initial_context = _build_context_from_chunks(chunks, max_chars=MAX_CHARS)
                    st.session_state.messages.append(
                        {
                            "role": "system",
                            "content": (
                                SYSTEM_PROMPT_QA
                                + "\n\n"
                                + DOCUMENT_BLOCK_TEMPLATE.format(
                                    document=initial_context
                                )
                            ),
                        }
                    )
                    if len(chunks) > 1:
                        st.success(
                            f"✅ Đã trích xuất {len(extracted):,} ký tự từ **{uploaded.name}** "
                            f"(chia thành {len(chunks)} phần để xử lý)."
                        )
                    else:
                        st.success(
                            f"✅ Đã trích xuất {len(extracted):,} ký tự từ **{uploaded.name}**."
                        )
            elif uploaded.type == "text/plain" or name_lower.endswith(".txt"):
                extracted = read_txt(uploaded)
                # Store full text (no truncation for chunking)
                st.session_state.document_text = extracted
                # Create smart chunks
                chunks = split_into_chunks(extracted)
                st.session_state.document_chunks = chunks
                st.session_state.file_name = uploaded.name
                st.session_state.summary = ""
                st.session_state.messages = []
                # Use first chunk(s) for initial context
                initial_context = _build_context_from_chunks(chunks, max_chars=MAX_CHARS)
                st.session_state.messages.append(
                    {
                        "role": "system",
                        "content": (
                            SYSTEM_PROMPT_QA
                            + "\n\n"
                            + DOCUMENT_BLOCK_TEMPLATE.format(
                                document=initial_context
                            )
                        ),
                    }
                )
                if len(chunks) > 1:
                    st.success(
                        f"✅ Đã trích xuất {len(extracted):,} ký tự từ **{uploaded.name}** "
                        f"(chia thành {len(chunks)} phần để xử lý)."
                    )
                else:
                    st.success(
                        f"✅ Đã trích xuất {len(extracted):,} ký tự từ **{uploaded.name}**."
                    )
            else:
                st.error("❌ Định dạng tệp không được hỗ trợ. Vui lòng tải lên PDF hoặc TXT.")
        except Exception as exc:  # noqa: BLE001
            st.error(f"❌ Không thể đọc tệp: {exc}")
            st.session_state.document_text = ""
            st.session_state.file_name = ""


has_doc = bool(st.session_state.document_text)
has_key = bool(api_key)
chunks = st.session_state.get("document_chunks", [])

if has_doc:
    chunk_info = f" ({len(chunks)} phần)" if len(chunks) > 1 else ""
    st.info(
        f"📚 Tài liệu hiện tại: **{st.session_state.file_name}** "
        f"({len(st.session_state.document_text):,} ký tự{chunk_info})."
    )
else:
    st.warning("📭 Chưa có tài liệu nào được tải lên.")


# ---------------------------------------------------------------------------
# Tabs: Summary / Q&A
# ---------------------------------------------------------------------------

tab_summary, tab_qa = st.tabs(["📝 Tóm tắt nhanh", "💬 Hỏi đáp với tài liệu"])


# ---- Tab 1: Tóm tắt nhanh ------------------------------------------------

with tab_summary:
    st.markdown("### Tóm tắt tài liệu")
    st.markdown(
        "Nhấn nút bên dưới để tạo bản tóm tắt gồm **Tổng quan**, "
        "**Các điểm chính** và **Kết luận**."
    )
    
    # Show chunk info if document has multiple chunks
    if len(chunks) > 1:
        st.caption(f"📄 Tài liệu được chia thành {len(chunks)} phần để xử lý tối ưu.")

    summarize_clicked = st.button(
        "✨ Tóm tắt tài liệu",
        disabled=not has_doc or not has_key,
        type="primary",
        use_container_width=True,
    )

    if summarize_clicked:
        if not has_doc:
            st.warning("Vui lòng tải lên tài liệu trước khi tóm tắt.")
        elif not has_key:
            st.warning(f"Vui lòng cung cấp {provider_name} API Key.")
        else:
            doc_chunks = st.session_state.get("document_chunks", [])
            
            if len(doc_chunks) <= 1:
                # Single chunk - simple summary
                prompt = build_summary_prompt(st.session_state.document_text)
                with st.spinner("Đang tạo tóm tắt..."):
                    try:
                        response = create_chat_completion(
                            prompt,
                            model=st.session_state.model,
                            temperature=0.3,
                            stream=False,
                            system_prompt="Bạn là trợ lý tóm tắt tài liệu chuyên nghiệp, trả lời bằng Markdown rõ ràng."
                        )
                        
                        if provider == "anthropic":
                            summary_text = response.content[0].text
                        elif provider == "google":
                            summary_text = response.text
                        else:
                            summary_text = response.choices[0].message.content or ""
                        
                        st.session_state.summary = summary_text
                    except Exception as exc:  # noqa: BLE001
                        st.error(handle_api_error(exc, provider))
            else:
                # Multi-chunk - summarize each then combine
                with st.spinner(f"Đang tóm tắt {len(doc_chunks)} phần..."):
                    try:
                        chunk_summaries = []
                        progress_bar = st.progress(0)
                        
                        for i, chunk in enumerate(doc_chunks):
                            progress_bar.progress((i + 1) / len(doc_chunks))
                            
                            # Summarize each chunk
                            chunk_prompt = (
                                f"TĐọc phần {i+1}/{len(doc_chunks)} của tài liệu sau và tạo tóm tắt ngắn gọn "
                                "(3-5 câu) về nội dung chính của phần này:\n\n"
                                f"--- PHẦN {i+1}/{len(doc_chunks)} ---\n"
                                f"{chunk['text'][:8000]}\n"
                                "--- HẾT PHẦN ---"
                            )
                            
                            response = create_chat_completion(
                                chunk_prompt,
                                model=st.session_state.model,
                                temperature=0.3,
                                stream=False,
                                system_prompt="Bạn là trợ lý tóm tắt tài liệu. Trả lời ngắn gọn, đi thẳng vào vấn đề."
                            )
                            
                            if provider == "anthropic":
                                chunk_sum = response.content[0].text
                            elif provider == "google":
                                chunk_sum = response.text
                            else:
                                chunk_sum = response.choices[0].message.content or ""
                            
                            chunk_summaries.append(f"**Phần {i+1}:** {chunk_sum}")
                        
                        # Now combine all summaries
                        progress_bar.progress(0.95)
                        combined_summaries = "\n\n".join(chunk_summaries)
                        
                        final_prompt = (
                            "Dưới đây là tóm tắt của các phần trong một tài liệu dài. "
                            "Hãy tạo một bản tóm tắt tổng hợp hoàn chỉnh với đúng ba phần:\n\n"
                            "## Tổng quan\n(Mô tả ngắn gọn nội dung tổng thể của tài liệu)\n\n"
                            "## Các điểm chính\n(Liệt kê các điểm chính dưới dạng danh sách gạch đầu dòng)\n\n"
                            "## Kết luận\n(Tóm tắt kết luận hoặc thông điệp cốt lõi của tài liệu)\n\n"
                            "--- TÓM TẮT CÁC PHẦN ---\n"
                            f"{combined_summaries}\n"
                            "--- HẾT TÓM TẮT ---"
                        )
                        
                        response = create_chat_completion(
                            final_prompt,
                            model=st.session_state.model,
                            temperature=0.3,
                            stream=False,
                            system_prompt="Bạn là trợ lý tóm tắt tài liệu chuyên nghiệp. Tạo tóm tắt rõ ràng, mạch lạc từ các phần tóm tắt được cung cấp."
                        )
                        
                        if provider == "anthropic":
                            summary_text = response.content[0].text
                        elif provider == "google":
                            summary_text = response.text
                        else:
                            summary_text = response.choices[0].message.content or ""
                        
                        progress_bar.empty()
                        st.session_state.summary = summary_text
                        
                    except Exception as exc:  # noqa: BLE001
                        st.error(handle_api_error(exc, provider))

    if st.session_state.summary:
        with st.container():
            st.markdown("---")
            st.markdown("#### Kết quả tóm tắt")
            st.markdown(st.session_state.summary, unsafe_allow_html=False)


# ---- Tab 2: Hỏi đáp ------------------------------------------------------

with tab_qa:
    st.markdown("### Hỏi đáp với tài liệu")
    st.markdown(
        "Đặt câu hỏi về nội dung tài liệu. Mô hình sẽ trả lời dựa trên văn bản đã tải lên."
    )
    
    # Show chunk info for Q&A
    if len(chunks) > 1:
        st.caption(f"💡 Tài liệu có {len(chunks)} phần - hệ thống sẽ tìm đoạn liên quan nhất để trả lời.")

    # Render chat history, skipping the hidden system message(s).
    for msg in st.session_state.messages:
        if msg.get("role") == "system":
            continue
        avatar = "👤" if msg["role"] == "user" else "🤖"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    user_question = st.chat_input(
        "Đặt câu hỏi về tài liệu...",
        disabled=not has_doc or not has_key,
    )

    if user_question:
        if not has_doc:
            st.warning("Vui lòng tải lên tài liệu trước khi đặt câu hỏi.")
            st.stop()
        if not has_key:
            st.warning(f"Vui lòng cung cấp {provider_name} API Key.")
            st.stop()

        # Append user turn and render it.
        st.session_state.messages.append(
            {"role": "user", "content": user_question}
        )
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_question)

        # Find relevant chunks for this question
        relevant_chunks = find_relevant_chunks(chunks, user_question, top_k=3)
        
        # Build context from relevant chunks
        if relevant_chunks:
            relevant_text = "\n\n---\n\n".join([c['text'] for c in relevant_chunks])
        else:
            relevant_text = st.session_state.document_text[:MAX_CHARS]
        
        # Build messages payload with updated context
        payload = [
            {
                "role": "system",
                "content": (
                    SYSTEM_PROMPT_QA
                    + "\n\n"
                    + DOCUMENT_BLOCK_TEMPLATE.format(document=relevant_text)
                ),
            }
        ]
        # Add conversation history (excluding old system messages)
        for msg in st.session_state.messages:
            if msg.get("role") != "system":
                payload.append({"role": msg["role"], "content": msg["content"]})

        try:
            if provider == "google":
                # Google Gemini handles streaming differently
                combined_content = [SYSTEM_PROMPT_QA + "\n\n" + DOCUMENT_BLOCK_TEMPLATE.format(document=relevant_text)]
                for msg in st.session_state.messages:
                    if msg.get("role") != "system":
                        role = "User" if msg["role"] == "user" else "Assistant"
                        combined_content.append(f"{role}: {msg['content']}")
                combined_content.append("Assistant: ")
                full_prompt = "\n\n".join(combined_content)
                
                model_gemini = google_client.generate_model(st.session_state.model)
                response = model_gemini.generate_content(
                    full_prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.3,
                        stream=True,
                    )
                )
                
                with st.chat_message("assistant", avatar="🤖"):
                    assistant_reply = st.write_stream(
                        chunk.text for chunk in response
                    )
                    
            elif provider == "anthropic":
                # Anthropic streaming
                anthropic_messages = []
                for msg in payload:
                    if msg["role"] == "system":
                        continue
                    anthropic_messages.append({
                        "role": "user" if msg["role"] == "user" else "assistant",
                        "content": msg["content"]
                    })
                
                with st.chat_message("assistant", avatar="🤖"):
                    with anthropic_client.messages.stream(
                        model=st.session_state.model,
                        max_tokens=4096,
                        messages=anthropic_messages,
                        temperature=0.3,
                    ) as stream:
                        assistant_reply = stream.get_full_content()
                assistant_reply = assistant_reply.content[0].text if hasattr(assistant_reply, 'content') else str(assistant_reply)
                
            else:
                # Groq and OpenAI use standard chat.completions
                stream = client.chat.completions.create(
                    model=st.session_state.model,
                    messages=payload,
                    temperature=0.3,
                    stream=True,
                    max_tokens=1024,  # Limit response size
                )
                with st.chat_message("assistant", avatar="🤖"):
                    assistant_reply = st.write_stream(
                        (chunk.choices[0].delta.content or "")
                        for chunk in stream
                    )
            
            st.session_state.messages.append(
                {"role": "assistant", "content": assistant_reply or ""}
            )
        except Exception as exc:  # noqa: BLE001
            st.error(handle_api_error(exc, provider))
            # Remove the just-appended user turn so it can be retried.
            if (
                st.session_state.messages
                and st.session_state.messages[-1].get("role") == "user"
            ):
                st.session_state.messages.pop()


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown("---")
st.caption(
    f"Xây dựng với ❤️ bằng Streamlit + {provider_name}. "
    "Tài liệu được xử lý cục bộ — chỉ gửi nội dung đến API khi bạn yêu cầu."
)