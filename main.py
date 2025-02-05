
from openai import OpenAI
# from fastapi import FastAPI
import requests
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
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
 You are tasked with analyzing a transcript of an online conversation between tech support and a customer. Use 'fetch_discord_messages' to get the conversation as a JSON object.
 Your analysis should focus on three aspects: a concise summary of the entire conversation, customer sentiment and troubleshooting difficulty. 
 Below is a conversation transcript. This transcript includes customer's issue description, agent's responses, and a possible resolution or status at the end of the conversation. 
  
 Keep in mind the following: 
 1. When summarizing the conversation, include a description the problem, the symptoms experienced, and the steps taken to try and resolve it. 
 2. Overall Sentiment and Difficulty Evaluation: Assess both sentiment and troubleshooting difficulty based on the entire conversation, focusing primarily on the customer's responses. 
 3. Conversation Length and Clarity: When evaluating difficulty, consider the conversation's length and the clarity of the customer's responses in aiding troubleshooting. 
 4. Key Phrases Identification: Identify phrases indicating satisfaction ("thank you, that solved my issue") or dissatisfaction ("this is still not working") to gauge sentiment. 
 5. Perspective Balance: Focus primarily on the customer's sentiment but also consider the agent's responses and their impact on the conversation's tone. 
 6. Adaptability to Variability: Be prepared to handle variations in conversation length, complexity, and subject matter. 
  
 <transcript> 
 {transcript} 
 </transcript> 
  
 Return a JSON response. Remember, a 0 score is negative, 50 is neutral and 100 is positive. 
 """


model = OpenAIModel('gpt-4o', api_key=OPENAI_API_KEY)
sentiment_agent = Agent(
    model=model,
    deps_type=CoversationObject,
    system_prompt=prompt_template_str,
    result_type=SummarizedOutput
)

# @sentiment_agent.tool


def fetch_discord_messages(discord_id: str) -> dict:
    url = f"http://134.195.91.7:5005/discord/{discord_id}/get_messages"
    response = requests.get(url, auth=HTTPBasicAuth(USERNAME, PASSWORD))

    if response.status_code == 200:  # Check if the request was successful
        data = response.json()  # Parse JSON response
        conv_obj = CoversationObject(**data)
        result = proccess_message(conv_obj)
        return result
    else:
        return {"Error Code: ": response.status_code}


def proccess_message(data: CoversationObject):
    try:
        # Create a mapping of author_id to author name
        author_map = {msg["author_id"]: msg["author"] for msg in data.messages}
        # Regex pattern to match <@author_id>

        author_tag_pattern = re.compile(r"<@([0-9]+)>")

        for msg in data.messages:
            if msg["content"]:
                # Replace all occurrences of author tags in content
                msg["content"] = author_tag_pattern.sub(
                    lambda m: author_map.get(m.group(1), m.group(0)), msg["content"])

        return data
    except Exception as e:
        print("Error processing messages: ", e)


def llm():
    pass


dict_data = fetch_discord_messages(1333653475753721936)
result = sentiment_agent.run_sync(
    "What is the sentiment of this conversation?", deps=dict_data)
print(result.data)
