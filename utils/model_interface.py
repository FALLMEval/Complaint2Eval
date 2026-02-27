"""
Model interface module for calling language models via OpenRouter
"""

import asyncio
import os
from time import sleep
from typing import Dict, List, Optional
from datetime import datetime
from openai import AsyncOpenAI, OpenAI  # 导入 AsyncOpenAI


class ModelInterface:
    """Simple interface for OpenRouter API calls"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("API key required")

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )
        self.async_client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )

    def call_model(
        self,
        model: str,
        user_message: str,
        temperature: float = 0.1,
        max_retries: int = 3,
    ) -> str:
        """Call model with retry logic"""
        for attempt in range(max_retries):
            try:
                completion = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": user_message}],
                    temperature=temperature,
                )
                return completion.choices[0].message.content

            except Exception as e:
                if attempt < max_retries - 1:
                    sleep(2**attempt)
                    continue
                return f"[Error] {str(e)}"

        return "[Error] Max retries exceeded"

    async def call_model_asyncio(
        self,
        model: str,
        user_message: str,
        temperature: float = 0.1,
    ) -> str:
        """Call model with retry logic"""
        try:
            completion = await self.async_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": user_message}],
                temperature=temperature,
            )
            return completion.choices[0].message.content

        except Exception as e:
            print(f"Error calling model {model}: {e}")
            return f"[Error] {str(e)}"

    async def batch_call(
        self,
        prompts: List[str],
        models: List[str],
        temperature: float = 0,
        max_concurrent: int = 10,
    ) -> List[Dict[str, str]]:
        """
        Batch async calls for multiple prompts and models combinations

        Args:
            prompts: List of prompts
            models: List of models
            temperature: Temperature parameter
            max_concurrent: Maximum concurrent requests

        Returns:
            List[Dict]: [{"prompt": str, "model": str, "response": str}, ...]
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_one(prompt: str, model: str) -> Dict[str, str]:
            async with semaphore:
                response = await self.call_model_asyncio(model, prompt, temperature)
                return {
                    "prompt": prompt,
                    "model": model,
                    "response": response,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }

        tasks = [process_one(prompt, model) for prompt in prompts for model in models]
        results = await asyncio.gather(*tasks)
        return results


# Standalone function for direct use
def call_openrouter_model(
    model: str,
    user_message: str,
    temperature: float = 0.1,
    api_key: Optional[str] = None,
) -> str:
    """
    Call OpenRouter model directly

    Parameters:
        model (str): Model name (e.g., "openai/gpt-4o")
        user_message (str): Prompt to send
        temperature (float): Temperature for response generation (default: 0.1)
        api_key (str): Optional API key

    Returns:
        str: Model response
    """
    try:
        interface = ModelInterface(api_key)
        return interface.call_model(model, user_message, temperature)
    except Exception as e:
        return f"[Error] {str(e)}"
