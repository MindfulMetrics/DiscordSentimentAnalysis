
from openai import OpenAI
# from fastapi import FastAPI
import requests
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.settings import ModelSettings
from pydantic_ai.models.openai import OpenAIModel
from sentiment_agent_classes import CoversationObject, SummarizedOutput
import os
import re

# Load the .env file
load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
USERNAME = os.getenv('API_USERNAME')
PASSWORD = os.getenv('API_PASSWORD')

prompt_template_str = """
You are a real-time sentiment analysis assistant focused on troubleshooting forums. Your task is to analyze an ongoing conversation between a customer experiencing product issues and a technician providing support. The goal is to assess the sentiment throughout the troubleshooting process and detect any major dissatisfaction or severe frustration aimed at the technician that may require escalation.

1. Real-Time Conversation Summary:
    * Provide a concise summary of the conversation, highlighting the customer’s reported problem, the symptoms experienced, and the troubleshooting steps or advice provided by the technician.

2. Overall Troubleshooting Sentiment Analysis:
    * Evaluate the sentiment of the entire conversation, with special emphasis on the customer's responses.
    * Identify any key phrases that indicate severe frustration or dissatisfaction (e.g., “I can’t believe this isn’t working”, “your advice is useless”, “I’m fed up with these issues”)—particularly if they target the technician.
    * Assign an overall sentiment score on a scale from 0 to 100 (0 = very negative, 50 = neutral, 100 = very positive).

3. Final Message Sentiment:
    * Assess the sentiment of the most recent (final) customer message in the conversation.
    * Provide a sentiment score on the same 0 to 100 scale along with a brief explanation.

4. Escalation Determination:
    * Determine whether the conversation exhibits signs of major dissatisfaction or severe frustration directed toward the technician.
    * Flag the conversation for escalation if the overall sentiment score is critically low (e.g., below 30) or if the final message explicitly expresses severe frustration with the technician.
    
5. Output Format:
    * Return your results as a JSON object that adheres to the schema provided. Ensure that all reasoning is clear and concise to support fast decision-making.
"""


#model = OpenAIModel('gpt-4o-mini', api_key=OPENAI_API_KEY)

model = OpenAIModel(
    'mistralai/ministral-8b',
    base_url='https://openrouter.ai/api/v1',
    api_key='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',
)

settings = ModelSettings(temperature=0)

sentiment_agent = Agent(
    model=model,
    deps_type=CoversationObject,
    system_prompt=prompt_template_str,
    result_type=SummarizedOutput
)

# @sentiment_agent.tool
    
def parse_discord_messages(discord_id: str) -> str:
    url = f"http://134.195.91.7:5005/discord/{discord_id}/get_messages"
    response = requests.get(url, auth=HTTPBasicAuth(USERNAME, PASSWORD))

    conversation = ""
    if response.status_code == 200:  # Check if the request was successful
        data = response.json()  # Parse JSON response
        conversation += f"Thread: {data['title']}\n\n"
        for message in data['messages']:
            if message['content'] == "":
                continue
            if message['is_customer']:
                conversation += "[Customer]: "
            elif message['author'] == "Marcus":
                conversation += "[AI Chatbot]: "
            else:
                conversation += "[Technician]: "
            conversation += f"{message['content']}\n"
        return conversation
    else:
        return {"Error Code: ": response.status_code}


transcript = parse_discord_messages(1318963547883049010)
result = sentiment_agent.run_sync(
f"""Carefully analyze the sentiment of this conversation:
<conversation>
{transcript} 
</conversation>
""", model_settings=settings)
print(result.data)
