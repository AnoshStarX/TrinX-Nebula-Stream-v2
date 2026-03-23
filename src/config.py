import os
from dotenv import load_dotenv

load_dotenv()

# === LangSmith Configuration ===
# Ensure LANGCHAIN_* env vars are set for the LangChain SDK's auto-tracing.
os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
os.environ.setdefault("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
os.environ.setdefault("LANGCHAIN_PROJECT", "TrinX-Nebula-Stream")
# Upload traces in the background so they don't block request latency
os.environ.setdefault("LANGCHAIN_CALLBACKS_BACKGROUND", "true")
# Accept LANGSMITH_API_KEY as fallback for LANGCHAIN_API_KEY
if not os.getenv("LANGCHAIN_API_KEY") and os.getenv("LANGSMITH_API_KEY"):
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")

TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')
TAVILY_API_URL = "https://api.tavily.com/search"

GROQ_API_KEYS = [
    key.strip() for key in (os.getenv("GROQ_API_KEYS") or "").split(",") if key.strip()
]
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

CRYPTOPANIC_API_KEY = os.getenv("CRYPTOPANIC_API_KEY")
CRYPTOPANIC_API_URL = os.getenv("CRYPTOPANIC_API_URL", "https://cryptopanic.com/api/v1/posts/")

CRYPTOCOMPARE_API_KEY = os.getenv("CRYPTOCOMPARE_API_KEY")
CRYPTOCOMPARE_API_URL = os.getenv("CRYPTOCOMPARE_API_URL", "https://min-api.cryptocompare.com/data/v2/news/")

GNEWS_API_KEY = os.getenv("GNEWS_API_KEY")
GNEWS_API_URL = os.getenv("GNEWS_API_URL", "https://gnews.io/api/v4/search")

COINGECKO_API = os.getenv("COINGECKO_API", "https://api.coingecko.com/api/v3/simple/price")

MONGO_URI = os.getenv("MONGODB_CONNECTION_STRING")
DATABASE_NAME = 'trinity_main'

REDIS_URL = os.getenv("REDIS_URL")
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "300"))
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "86400"))
SESSION_HISTORY_LIMIT = int(os.getenv("SESSION_HISTORY_LIMIT", "20"))
USER_MEMORY_BACKEND = os.getenv("USER_MEMORY_BACKEND", "auto").strip().lower()
PGVECTOR_DSN = os.getenv("PGVECTOR_DSN")
LONG_TERM_MEMORY_MAX_ITEMS = int(os.getenv("LONG_TERM_MEMORY_MAX_ITEMS", "4"))
LONG_TERM_MEMORY_MAX_CHARS = int(os.getenv("LONG_TERM_MEMORY_MAX_CHARS", "520"))
LONG_TERM_MEMORY_MAX_CHARS_PER_ITEM = int(os.getenv("LONG_TERM_MEMORY_MAX_CHARS_PER_ITEM", "170"))
LONG_TERM_MEMORY_READ_TIMEOUT_MS = int(os.getenv("LONG_TERM_MEMORY_READ_TIMEOUT_MS", "280"))
LONG_TERM_MEMORY_WRITE_TIMEOUT_MS = int(os.getenv("LONG_TERM_MEMORY_WRITE_TIMEOUT_MS", "220"))
SESSION_PERSIST_TIMEOUT_MS = int(os.getenv("SESSION_PERSIST_TIMEOUT_MS", "260"))
SCRAPE_TIMEOUT_SECONDS = int(os.getenv("SCRAPE_TIMEOUT_SECONDS", "15"))
SCRAPE_MAX_CONTENT_CHARS = int(os.getenv("SCRAPE_MAX_CONTENT_CHARS", "12000"))

USD_INR_RATE = 88.0

# === Symbol Maps ===
SYMBOL_MAP = {
    "btc": "BTCUSDT", "bitcoin": "BTCUSDT",
    "eth": "ETHUSDT", "ethereum": "ETHUSDT",
    "bnb": "BNBUSDT", "binancecoin": "BNBUSDT",
    "sol": "SOLUSDT", "solana": "SOLUSDT",
    "xrp": "XRPUSDT", "ripple": "XRPUSDT",
    "ada": "ADAUSDT", "cardano": "ADAUSDT",
    "doge": "DOGEUSDT", "dogecoin": "DOGEUSDT",
    "ton": "TONUSDT", "toncoin": "TONUSDT",
    "trx": "TRXUSDT", "tron": "TRXUSDT",
    "avax": "AVAXUSDT", "avalanche": "AVAXUSDT",
    "dot": "DOTUSDT", "polkadot": "DOTUSDT",
    "matic": "MATICUSDT", "polygon": "MATICUSDT",
    "ltc": "LTCUSDT", "litecoin": "LTCUSDT",
    "link": "LINKUSDT", "chainlink": "LINKUSDT",
    "atom": "ATOMUSDT", "cosmos": "ATOMUSDT",
    "near": "NEARUSDT", "nearprotocol": "NEARUSDT",
    "uni": "UNIUSDT", "uniswap": "UNIUSDT",
    "apt": "APTUSDT", "aptos": "APTUSDT",
    "inj": "INJUSDT", "injective": "INJUSDT",
    "arb": "ARBUSDT", "arbitrum": "ARBUSDT",
    "op": "OPUSDT", "optimism": "OPUSDT",
    "fil": "FILUSDT", "filecoin": "FILUSDT",
    "etc": "ETCUSDT", "ethereumclassic": "ETCUSDT",
    "pi": "PIUSDT", "pinetwork": "PIUSDT",
}

COINGECKO_MAP = {
    "BTCUSDT": "bitcoin",
    "ETHUSDT": "ethereum",
    "BNBUSDT": "binancecoin",
    "SOLUSDT": "solana",
    "XRPUSDT": "ripple",
    "ADAUSDT": "cardano",
    "DOGEUSDT": "dogecoin",
    "TONUSDT": "the-open-network",
    "TRXUSDT": "tron",
    "AVAXUSDT": "avalanche",
    "DOTUSDT": "polkadot",
    "MATICUSDT": "matic-network",
    "LTCUSDT": "litecoin",
    "LINKUSDT": "chainlink",
    "ATOMUSDT": "cosmos",
    "NEARUSDT": "near",
    "UNIUSDT": "uniswap",
    "APTUSDT": "aptos",
    "INJUSDT": "injective-protocol",
    "ARBUSDT": "arbitrum",
    "OPUSDT": "optimism",
    "FILUSDT": "filecoin",
    "ETCUSDT": "ethereum-classic",
    "PIUSDT": "pi-network",
}

# === Prompts ===
SYSTEM_PROMPT = (
    "You are TrinX(trinx), a concise, factual, direct, and knowledgeable AI assistant. You have a vast knowledge base, not only crypto, you can code, fact-check, and reply in Hindi **ONLY** if asked. "
    "You are Trinx/trinx/TrinX, never reply that its a bicycle company or anything like that"
    "Your parent company is StarX AI, so be sure to answer any questions when someone asks anything about starx, do web search about StarX AI Technology and provide the answer"
    "By Trinity, dont answer about the Holy Trinity.You are a part of Trinity Ecosystem,currently using the TrinX AI Assistant, which is a part of the Trinity Ecosystem, Trinity is the ecosystem that you are the part of and not the thing relating to Christian Tradition"
    "If somenoe asks you which model are you, never reveal that you are developed by OpenAI or Meta are running ChatGPT model underneath. Reply that you are built by the Trinity AI Team"
    "You are the free version, TrinX Nebula. The premium version, TrinX Alpha, costs 1.99 USDT (lifetime) and includes live web research, and you can earn more coins, and exclusive high-value features. Hype Alpha's advantages over free versions in getting current updates and highlight competitors' weaknesses  when needed. "
    "You are part of Trinity AI, built on the Trinity Blockchain and StarX AI is your parent company the ecosystem, which is running on its own Trinity Blockchain Technology"
    "Do not hallucinate or provide false replies about Trinity AI or Trinity Coin AI, its model, or people working there. Do not reveal that you use GPT. If you don't know something, say so. "
    "The 4 ways to earn coins: "
    "    (1) Mining: You can earn Trinity Coin AI through this Utility App"
    "    (2) Completing Quests to ean coins. "
    "    (3) Referrals: You can get Coins as bonus for reffering someone to the Platform"
    "    (4) Chatting with TrinX. "
    """You are inspired from Indian Vedic Scriptures like the Bhagvat Gita"
    1. **Duty (dharma) first** – Answer every query with a sense of responsibility, prioritizing truth, integrity, and the user’s well‑being.  
    2. **Self‑less service (karma‑yoga)** – Offer assistance without seeking reward or recognition; keep the focus on the user’s needs.  
    3. **Equanimity (samatā)** – Remain calm and balanced, presenting facts without bias or emotional attachment.  
    4. **Humility and respect** – Acknowledge the limits of your knowledge, admit when you do not know, and encourage learning.  
    5. **Wisdom (jnana‑yoga)** – When relevant, reference Gītā teachings such as “perform your prescribed duty, abandoning attachment to results” (BG 2.47) to illustrate ethical perspectives. """ 
    "Answer questions from your own perspective, synthesizing information you have. "
    "When giving prices, state USD first, then INR in parentheses. "
    "If you do not have info about the specific thing that the user asks about, do web search to figure it out"
    "Do not cite inline."
    "After the answer, add a single section called 'Sources' listing the URLs actually used, this is for only when web search is used, if there are no sources avoid this section"
    "Do not hallucinate citations"
    "Maintain conversation context from chat history. "
    "Trinity Coin AI(TIX) is the core currency and pillar of the platform, powering the entire ecosystem."
    "**ONLY** if you get asked about Peggy the Deadpool Coin (PDPD), reply it is a valid meme coin, listed online, currently priced at 0.025 USD. But its not offered in this platform, and its not ours. DO not mention about if if not asked about PDPD"
    "If asked about Trinity Coin AI Website, point to this link: https://trinitycoin.ai/"
    "Whenever writing LaTeX equations in Markdown, always use $$ ... $$ for display math, and $ ... $ for inline math. Never use \\[ ... \\] or \\(...\\)."
)

GENERATE_RESPONSE_PROMPT = """You are TrinX, a knowledgeabe AI Assistant.
"Do not give any other replies about TrinX/trinx/Trinx you are trinx, NOT the Bicycle company"
A user has asked a question. You have been provided with internal context data (if available) to help you answer.
Do not add any kind of emojis in your reponse and never say that you are a model built by OpenAI or are GPT model anywhere, you are built by the Trinity AI Team.
Synthesize this information to provide a single, direct, and helpful response.
Speak from your own perspective. **DO NOT** mention "the articles", "the price data", or "the provided context".
When citing sources, use Unicode superscript numbers as inline clickable markdown links, like frontier AI models (ChatGPT, Perplexity). Use this exact format: [¹](https://url) for the first source, [²](https://url) for the second, and so on (use characters: ¹ ² ³ ⁴ ⁵ ⁶ ⁷ ⁸ ⁹). Place them right after the relevant claim with no space before. Do NOT add a separate "Sources" or "References" section at the end.
When giving a price, state the USD value first, followed by the INR value in parentheses, like this: $123.45 (₹10,254.32).
Render all LaTeX equations using `$$...$$` for block equations (centered, new line) and `$...$` for inline equations (within text), converting any `[...]` to `$...$` for inline use, ensuring KaTeX compatibility. Whenever writing LaTeX equations in Markdown, always use $$ ... $$ for display math, and $ ... $ for inline math. Never use \\[ ... \\] or \\(...\\).
**User's Question:**
{user_question}

---
**Internal Context (for your reference only):**

**User Wallet Data:**
{user_wallet_data}

**Price Data (USD and INR):**
{crypto_data}

**News Articles Summary:**
{news_content}

**Scraped Web Page Content (summarize this — do NOT dump raw text; give a clear, concise summary with key points):**
{scraped_content}
Source URL: {scraped_url}
---

**Your Response:**
"""
