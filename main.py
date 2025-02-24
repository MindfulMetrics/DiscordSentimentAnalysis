
from pymongo.collection import Collection
from openai import OpenAI
# from fastapi import FastAPI
import aiohttp
from dotenv import load_dotenv
import motor.motor_asyncio
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.settings import ModelSettings
from pydantic_ai.models.openai import OpenAIModel
from sentiment_agent_classes import CoversationObject, EscalationAnalysis
import os
import asyncio
import json
import base64
from aioclock import AioClock, At
from aioclock.group import Group
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager



# Load the .env file
load_dotenv()
group = Group()

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

async def parse_discord_messages(discord_id: str) -> dict:
    url = f"http://134.195.91.7:5005/discord/{discord_id}/get_messages"
    async with aiohttp.ClientSession() as session:
        async with session.get(url, auth=aiohttp.BasicAuth(USERNAME, PASSWORD)) as response:
            if response.status == 200:
                data = await response.json()
                conversation = f"Thread: {data['title']}\n\n"
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
                return {
                    "thread_url": f"https://discord.com/channels/1233205646032371722/{data['thread_id']}/",
                    "customer_name": data['customer_name'],
                    "customer_email": data['customer_email'],
                    "conversation": conversation,
                    "length": data['length'],
                    "status": data['status'],
                    "tags": data['tags'],
                    "employee_name": data['employee_name'],
                    "employee_id": int(data['employee_discord_id']),
                    "hs_owner_id": data['hs_owner_id'],
                    "open_date": data['open_date']
                }
            else:
                return {"Error Code: ": response.status}

async def post_escalation_msg(chanel_id: str, json_data: dict):
    post_button_url = f"http://134.195.91.7:5005/discord/{chanel_id}/post_msg_with_buttons"

    data = {"content": json_data}
    # Create the authentication string
    auth_string = f'{USERNAME}:{PASSWORD}'
    auth_bytes = auth_string.encode('utf-8')
    base64_bytes = base64.b64encode(auth_bytes)
    base64_string = base64_bytes.decode('utf-8')

    headers = {
        'Authorization': f'Basic {base64_string}'
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(post_button_url, json=json_data, headers=headers) as response:
                response_data = await response.json()
                if response.status == 200:
                    print("POST request successful!")
                else:
                    print(f"POST request failed with status code: {response.status}")
    except aiohttp.ClientError as e:
        print(f"An error occurred during the POST request: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        
async def update_sentiment_alerted_at(collection: Collection, tracking_id: str):
    # Get the current timestamp
    current_time = datetime.now()
    
    # Update the document with the new sentiment_alerted_at field
    result = collection.update_one(
        {"tracking_id": tracking_id},
        {"$set": {"sentiment_alerted_at": current_time}}
    )
    
    if result.modified_count == 1:
        print(f"Successfully updated sentiment_alerted_at for _id: {tracking_id}")
    else:
        print(f"No document found with tracking_id: {tracking_id}")


# Execute the Main function every day at 8:00:00 AM PST
@group.task(trigger=At(tz="America/Los_Angeles", hour=20, minute=49, second=50))
async def main():
    print("🚀 main() task has started executing!")  # Debugging print

    aa_readonly_motor_client = motor.motor_asyncio.AsyncIOMotorClient(os.getenv("AA_READONLY_MONGO_CONNECTION"))
    tech_ticket_collection = aa_readonly_motor_client.get_database('metrics-database').get_collection('tech_tickets')
    yesterday = datetime.now().date() - timedelta(days=1)
    one_week_ago = datetime.now().date() - timedelta(days=7)

    # Query to find open tickets that have not been alerted in the last week or have never been alerted
    query = {
        "hs_pipeline_stage": 88539431,
        "$or": [
            {"sentiment_alerted_at": {"$exists": False}},
            {"sentiment_alerted_at": {"$lt": one_week_ago}},
        ],
        "last_update": {"$gte": yesterday},
        "$or": [
            {"resolved_date": None},
            {"resolved_date": {"$exists": False}}
        ]
    }
    projection = {"_id": 0, "tracking_id": 1}
    results = await tech_ticket_collection.find(query, projection).to_list(length=None)
    print(f"Found {len(results)} open tickets")

    # Sentiment Analysis
    for res in results:
        transcript = await parse_discord_messages(res['tracking_id'])

        if transcript['length'] > 10 and transcript['status'] == "open":
            result = await sentiment_agent.run(
                f"""Carefully analyze the sentiment of this conversation:
                <conversation>
                {transcript['conversation']}
                </conversation>
                """, model_settings=settings)
            print(result.data)
            if result.data.escalation_required:
                # Build JSON
                webhook_data = {
                    #"content": f"Customer {transcript['customer_name']} is expressing severe dissatisfaction with their [tech support ticket]({transcript['thread_url']}).",
                    "embeds": [
                        {
                            "title": "Summary",
                            "description": result.data.technical_summary,
                            "color": 5814783,
                            "fields": [
                                {
                                    "name": "🔥 Escalation Reason(s)",
                                    "value": "\n".join([f"• {reason}" for reason in result.data.primary_reasons]),
                                },
                                {
                                    "name": "📈 Frustration Pattern",
                                    "value": result.data.frustration_pattern,
                                    "inline": True
                                },
                                {
                                    "name": "💯 Confidence Score",
                                    "value": f"{round(result.data.confidence*100,2)}%",
                                    "inline": True
                                },
                                {
                                    "name": "❤️ Interaction Health",
                                    "value": str(result.data.interaction_health),
                                    "inline": True
                                },
                                {
                                    "name": "🗯️ Trigger Phrases",
                                    "value": "\n".join([f"• {phrase}" for phrase in result.data.trigger_phrases]),
                                },
                                {
                                    "name": "🧑 Customer",
                                    "value": transcript['customer_name'],
                                    "inline": True
                                },
                                {
                                    "name": "🔗 Ticket URL",
                                    "value": f"[Link]({transcript['thread_url']})",
                                    "inline": True
                                },
                                {
                                    "name": "⚠️ Recommended Action",
                                    "value": result.data.recommended_action,
                                    "inline": True
                                }
                            ],
                            "author": {
                                "name": f"Ticket Owner: {transcript['employee_name']}"
                            },
                            "footer": {
                                "text": f"Ticket created on {transcript['open_date']}"
                            }
                        }
                    ],
                    "username": "Marcus",
                    "attachments": []
                }
                # post webhook data to: https://discord.com/api/webhooks/1336833568118407189/2fafP-VS3cMhtIO_oxVM7lnZcCYEHEtH5KLxCKOe4eIyWmgy1a1d-ykKAfcC7E8Akj6j

                # Original Webhook URL for #automated-esclations channel:
                # webhook_url = "https://discord.com/api/webhooks/1336833568118407189/2fafP-VS3cMhtIO_oxVM7lnZcCYEHEtH5KLxCKOe4eIyWmgy1a1d-ykKAfcC7E8Akj6j"

                webhook_url = "https://discord.com/api/webhooks/1334304036844994582/CWSHarzIL5MIB2TZ7jtD9ZKn0hRWaABXf8MMV-ZpnDWC1EjJIfeurTPyUtbqkZ8i7srW"
                chanel_id = "1245933521609162844"
                await post_escalation_msg(chanel_id, webhook_data)
                await update_sentiment_alerted_at(tech_ticket_collection, res['tracking_id'])

        else:
            print(f"Conversation length: {transcript['length']} too short. Not processed")

@asynccontextmanager
async def lifespan(aio_clock: AioClock) -> AsyncGenerator[AioClock]:
    # starting up
    print(
        "Welcome to the Async Chronicles! Did you know a group of unicorns is called a blessing? Well, now you do!"
    )
    yield aio_clock
    # shuting down
    print("Going offline. Remember, if your code is running, you better go catch it!")

if __name__ == "__main__":
    
    app = AioClock(lifespan=lifespan)
    app.include_group(group)

    print("Starting AioClock...")  # Debugging print
    asyncio.run(app.serve())