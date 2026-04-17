"""Configuration module for Agentic RAG system"""

import os
import requests
from dotenv import load_dotenv

# ⚠️ Keeping your existing imports (as requested)
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain.chat_models import ChatOllama  # deprecated but retained


# Load environment variables
load_dotenv()


class Config:
    """Configuration class for RAG system"""

    # =========================
    # 🔑 API Keys
    # =========================
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    # =========================
    # 🤖 LLM Provider
    # =========================
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")

    # =========================
    # 🧠 Model Configurations
    # =========================
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # Ollama
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

    # LM Studio
    LMSTUDIO_BASE_URL = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
    LMSTUDIO_MODEL = os.getenv("LMSTUDIO_MODEL", "llama-3.2-3b-instruct")

    # =========================
    # 📦 Vector DB
    # =========================
    VECTOR_DB = os.getenv("VECTOR_DB", "azure")

    # =========================
    # 📄 Document Processing
    # =========================
    CHUNK_SIZE = 1000 #500
    CHUNK_OVERLAP = 200 #50

    # =========================
    # ⚙️ LLM Controls
    # =========================
    USE_LLM_FOR_GRAPH = True
    USE_LLM_FOR_ANSWER = True

    ZERO_LLM_MODE = not USE_LLM_FOR_ANSWER
    CONFIDENCE_THRESHOLD = 0.6

    # =========================
    # 🌐 Default URLs
    # =========================
    DEFAULT_URLS = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/"
    ]

    # =========================
    # 🔍 Health Check Helpers
    # =========================
    @staticmethod
    def _is_lmstudio_running():
        try:
            res = requests.get(f"{Config.LMSTUDIO_BASE_URL}/models", timeout=2)
            return res.status_code == 200
        except:
            return False

    @staticmethod
    def _is_ollama_running():
        try:
            res = requests.get(f"{Config.OLLAMA_BASE_URL}/api/tags", timeout=2)
            return res.status_code == 200
        except:
            return False

    # =========================
    # 🧠 LLM Factory with Fallback
    # =========================
    @staticmethod
    def get_llm():
        """Return LLM instance with fallback:
        LM Studio → Ollama → Azure → OpenAI
        """

        provider = Config.LLM_PROVIDER.lower()
        print(f"LLM Provider set to: {provider}")

        # =========================
        # 🎯 Priority-based selection
        # =========================

        # 1️⃣ Try LM Studio (if selected OR fallback mode)
        try:
            if provider == "lmstudio" or provider == "auto":
                if Config._is_lmstudio_running():
                    print("✅ Using LM Studio")

                    return ChatOpenAI(
                        base_url=Config.LMSTUDIO_BASE_URL,
                        api_key="lm-studio",
                        model=Config.LMSTUDIO_MODEL,
                        temperature=0,
                        timeout=120
                    )
                else:
                    print("⚠️ LM Studio not running")
        except Exception as e:
            print(f"❌ LM Studio failed: {e}")

        # 2️⃣ Try Ollama
        try:
            if provider in ["ollama", "auto", "lmstudio"]:
                if Config._is_ollama_running():
                    print("✅ Using Ollama")

                    return ChatOllama(
                        model=Config.OLLAMA_MODEL,
                        base_url=Config.OLLAMA_BASE_URL,
                        temperature=0
                    )
                else:
                    print("⚠️ Ollama not running")
        except Exception as e:
            print(f"❌ Ollama failed: {e}")

        # 3️⃣ Try Azure OpenAI
        try:
            if os.getenv("AZURE_OPENAI_ENDPOINT"):
                print("☁️ Using Azure OpenAI")

                return AzureChatOpenAI(
                    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                    api_key=os.getenv("AZURE_OPENAI_KEY"),
                    deployment_name=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
                    api_version="2024-02-01",
                    temperature=0,
                    timeout=60
                )
        except Exception as e:
            print(f"❌ Azure OpenAI failed: {e}")

        # 4️⃣ Fallback to OpenAI
        try:
            print("🌐 Falling back to OpenAI")

            return ChatOpenAI(
                api_key=Config.OPENAI_API_KEY,
                model=Config.OPENAI_MODEL,
                temperature=0,
                timeout=60
            )
        except Exception as e:
            print(f"❌ OpenAI failed: {e}")
            raise RuntimeError("No LLM provider available!")
