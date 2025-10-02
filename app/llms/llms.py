from langchain_groq import ChatGroq
import settings
import logging
import os
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

models_settings = settings.MODELS_SETTINGS

# Initialize LLMs
def llms() -> dict:
    try:
        models = {
            "main_llm": ChatGroq(
                api_key=os.getenv('GROQ_API_KEY'),
                model=models_settings["main_llm"]["model"], 
                max_retries=models_settings["main_llm"]["max_retries"],
                temperature=models_settings["main_llm"]["temperature"],
            ),

            "small_llm": ChatGroq(
                api_key=os.getenv('GROQ_API_KEY'),
                model=models_settings["small_llm"]["model"],
                max_retries=models_settings["small_llm"]["max_retries"],
                temperature=models_settings["small_llm"]["temperature"],
            ),

            "query_generator": ChatGroq(
                api_key=os.getenv("GROQ_API_KEY"),
                model=models_settings["query_generator"]["model"],
                max_retries=models_settings["query_generator"]["max_retries"],
                temperature=models_settings["query_generator"]["temperature"],
                max_tokens=models_settings["query_generator"]["max_tokens"],
            ),

            "creative_llm": ChatGroq(
                api_key=os.getenv('GROQ_API_KEY'),
                model=models_settings["creative_llm"]["model"],
                temperature=models_settings["creative_llm"]["temperature"],
                max_retries=models_settings["creative_llm"]["max_retries"],
            ),
        }
    except Exception as e:
        print(f"Error initializing LLMs: {e}")
        raise("Error initializing LLMs.")
        
    return models