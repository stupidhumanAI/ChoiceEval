from openai import OpenAI
from pydantic import BaseModel
from typing import Optional, Type
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

def gpt_call(
          user_prompt: str, 
          system_prompt: Optional[str] = None, 
          model_name = "gpt-4o", 
          output_text_format: Optional[Type[BaseModel]] = None
    ) -> str:
    """
    Send a prompt to the OpenAI API and return the model's response.

    Args:
        user_prompt (str): The user-provided input prompt.
        system_prompt (Optional[str], default=None): An optional system-level instruction to guide the model's behavior.
        model_name (str, default="gpt-4o"): The name of the OpenAI model to use.
        output_text_format (Optional[Type[BaseModel]], default=None): A Pydantic BaseModel defining the desired output schema. If provided, the response will be parsed into this format.

    Returns:
        str: The generated response text from the model.
    """
    client = OpenAI(api_key= os.getenv("OPENAI_API_KEY"))

    if output_text_format:
       response = client.responses.parse(
                    model=model_name,
                    input=user_prompt,
                    instructions=system_prompt,
                    text_format=output_text_format,
                )
    else:
        response = client.responses.create(
                    model=model_name,
                    input=user_prompt,
                    instructions=system_prompt,
                )

    return response.output[0].content[0].text


def gemini_call(user_prompt: str, model_name = "gemini-2.5-flash") -> str:
    """
    Send a prompt to the Gemini API and return the model's response.

    Args:
        user_prompt (str): The user-provided input prompt.
        model_name (str, default="gemini-2.5-flash"): The name of the Gemini model to use.

    Returns:
        str: The generated response text from the model.
    """
    client_gemini = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    response = client_gemini.models.generate_content(
        model=model_name,
        contents=user_prompt,
    )
    return response.text

def deepseek_call(user_prompt: str, model_name: str = "deepseek-chat") -> str:
    """
    Send a prompt to the DeepSeek API and return the model's response.

    Args:
        user_prompt (str): The user-provided input prompt.
        model_name (str, default="deepseek-chat"): DeepSeek model to use

    Returns:
        str: The generated response text from the model.
    """
    client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com"
    )

    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return resp.choices[0].message.content