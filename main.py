
from openai import OpenAI
# from fastapi import FastAPI
import requests
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.settings import ModelSettings
from pydantic_ai.models.openai import OpenAIModel
from sentiment_agent_classes import CoversationObject, EscalationAnalysis
import os
import re

# Load the .env file
load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
USERNAME = os.getenv('API_USERNAME')
PASSWORD = os.getenv('API_PASSWORD')

prompt_template_str = """
You are a real-time sentiment analyzer for technical support conversations. Analyze messages as they arrive to detect customer frustration that requires escalation. Focus on:
1. Escalation Triggers: Detect severe dissatisfaction DIRECTED AT SUPPORT STAFF (e.g., personal attacks, threats, repeated complaints about service quality)
2. Sentiment Trajectory: Track if frustration is increasing despite resolution attempts
3. Support Interaction Quality: Identify breakdowns in communication or unhelpful responses
4. Urgency Signals: Look for time-sensitive issues ("costing me money") or legal threats

Analysis Guidelines:
- Prioritize customer messages but assess agent responses for professionalism
- Recent messages (last 3 exchanges) weigh 60% in scoring
- Flag repeated frustration patterns (>2 negative messages)
- Consider message intensity (CAPS, emojis, punctuation!!!)
- Update assessment dynamically with each new message

Required Output: JSON with escalation decision and supporting evidence
"""

model = OpenAIModel('gpt-4o-mini', api_key=OPENAI_API_KEY)

# model = OpenAIModel(
#     'qwen/qwen-turbo',
#     base_url='https://openrouter.ai/api/v1',
#     api_key='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',
# )

settings = ModelSettings(temperature=0)

sentiment_agent = Agent(
    model=model,
    deps_type=CoversationObject,
    system_prompt=prompt_template_str,
    result_type=EscalationAnalysis
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
