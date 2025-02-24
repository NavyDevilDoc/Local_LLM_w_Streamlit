from langchain_openai import ChatOpenAI
from openai import AsyncOpenAI
import os
import io
from typing import Optional


class ModelInterface:
    def __init__(self):
        api_key = os.environ.get("OPENAI_API_KEY")
        
        # Initialize OpenAI models dictionary
        self.text_models = {
            "GPT-4o": ChatOpenAI(
                model_name="gpt-4o",
                temperature=0.1,
                api_key=api_key
            ),
            "GPT-4o-mini": ChatOpenAI(
                model_name="gpt-4o-mini",
                temperature=0.1,
                api_key=api_key
            )
        }
        # Initialize AsyncOpenAI client for DALL-E 3
        self.image_client = AsyncOpenAI(api_key=api_key)

    async def get_openai_response(self, prompt: str, model_name: str = "GPT-4o") -> Optional[str]:
        try:
            model = self.text_models.get(model_name)
            if not model:
                raise ValueError(f"Model {model_name} not found")
            
            # Create a system message to help format the response
            system_message = """You are a helpful AI assistant. Use the previous conversation history and context 
            to provide relevant and contextual responses. Maintain a consistent tone and remember details from 
            earlier in the conversation."""
            
            # Create the messages array with system message and user prompt
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ]
            
            response = await model.ainvoke(messages)
            return response.content
        except Exception as e:
            print(f"OpenAI Error: {e}")
            return None

    async def generate_image(self, prompt: str, size: str = "1024x1024", reference_image: io.BytesIO = None) -> Optional[str]:
        try:
            if reference_image:
                # Create variation with reference image
                response = await self.image_client.images.create_variation(
                    image=reference_image,
                    n=1,
                    size=size,     
                )
            else:
                # Generate new image
                response = await self.image_client.images.generate(
                    model="dall-e-3",
                    prompt=prompt,
                    size=size,
                    quality="standard",
                    n=1
                )
            
            return response.data[0].url
        except Exception as e:
            print(f"Image Generation Error: {str(e)}")
            return None
        
    async def generate_caption(self, image_url: str, model_name: str = "GPT-4o") -> Optional[str]:
        try:
            model = self.text_models.get(model_name)
            if not model:
                raise ValueError(f"Model {model_name} not found")
            
            system_message = """You are an expert at describing images. Provide clear, detailed, and accurate 
            captions that capture both the obvious and subtle elements of the image. Focus on the key elements 
            and overall composition."""
            
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"Please generate a detailed caption for this image: {image_url}"}
            ]
            
            response = await model.ainvoke(messages)
            return response.content
        except Exception as e:
            print(f"Caption Generation Error: {e}")
            return None