from ollama import Client
import ollama
from src.config import Config

from src.logger import Logger

logger = Logger()

client = Client(host=Config().get_ollama_api_endpoint())
print(client)

class Ollama:
    def __init__(self):
        try:
            self.client = ollama.Client(Config().get_ollama_api_endpoint())
            self.models = self.client.list()["models"]
            log.info("Ollama available")
        except:
            self.client = None
            log.warning("Ollama not available")
            log.warning("run ollama server to use ollama models otherwise use other models")

    def inference(self, model_id: str, prompt: str) -> str:
        try:
            response = client.chat(model=model_id, messages=[
            {
                'role': 'user',
                'content': prompt.strip(),
            },
            ])
            return response['message']['content']
        except Exception as e:
            logger.error(f"Error during model inference: {e}")
        return ""