import os
import asyncio
import base64
from datetime import datetime, timedelta, timezone
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
# from fastapi import FastAPI
import aiohttp
from dotenv import load_dotenv
import motor.motor_asyncio
from motor.motor_asyncio import AsyncIOMotorCollection
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
from pydantic_ai.providers.google import GoogleProvider
# from tqdm.asyncio import tqdm_asyncio
# from tqdm import tqdm
# pylint: disable=unused-import
# from aioclock import AioClock, At, Once, Every
# from aioclock.group import Group
from sentiment_agent_classes import CoversationObject, EscalationAnalysis

TEST_CHANNEL = "1245933521609162844"
AUTOMATED_ESCALATIONS_CHANNEL = "1336771513520492544"
LATENCY_WARNINGS_CHANNEL = "1341855677152231474"

# Load the .env file
load_dotenv()
# group = Group()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')  # Make sure this is set in your .env
USERNAME = os.getenv('API_USERNAME')
PASSWORD = os.getenv('API_PASSWORD')

PROMPT_TEMPLATE_STR = """
You are a real-time sentiment analyzer for technical support conversations. Analyze messages as they arrive to detect severe customer frustration that requires escalation to management or a supervisor. Focus on:
1. Escalation Triggers: Detect severe dissatisfaction DIRECTED AT SUPPORT STAFF (e.g., personal attacks, threats, threatening legal action, mention of lost money, repeated complaints about service quality)
2. Sentiment Trajectory: Track if frustration is increasing despite resolution attempts
3. Support Interaction Quality: Identify breakdowns in communication or unhelpful responses
4. Urgency Signals: Look for time-sensitive issues ("costing me money") or legal threats

Analysis Guidelines:
- Prioritize customer messages but assess technician responses for professionalism
- Recent messages (last 3 exchanges) weigh 60% in scoring
- Flag repeated frustration patterns (>2 negative messages)
- Consider message intensity (CAPS, emojis, punctuation!!!)
- Only escalate severe cases (e.g., threats, legal action). Confidence > 70% and multiple escalation factors present. Do not escalate for minor issues or customer dissatisfaction.
Required Output: JSON with escalation decision and supporting evidence
"""

provider = GoogleProvider(api_key=GEMINI_API_KEY)
model = GoogleModel('gemini-3-flash-preview', provider=provider)

# model = OpenAIModel(
#     'qwen/qwen-turbo',
#     base_url='https://openrouter.ai/api/v1',
#     api_key='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',
# )

settings = GoogleModelSettings(temperature=0)

sentiment_agent = Agent(
    model=model,
    model_settings=settings,
    deps_type=CoversationObject,
    system_prompt=PROMPT_TEMPLATE_STR,
    output_type=EscalationAnalysis
)

async def parse_discord_messages(discord_id: str) -> dict:
    """Fetch and parse Discord messages for a given user ID via API.
    
    Args:
        discord_id: Discord user ID to fetch messages for
    
    Returns:
        Dictionary containing parsed message data from API
    """
    url = f"http://134.195.91.7:5005/discord/{discord_id}/get_messages"
    if USERNAME is None or PASSWORD is None:
        raise ValueError("API_USERNAME and API_PASSWORD environment variables must be set.")
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

async def post_msg(chanel_id: str, json_data: dict):
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
                    return True
                else:
                    print(f"POST request failed with status code: {response.status}")
                    return True
    except aiohttp.ClientResponseError as e:
        print(f"Client response error occurred: {e}")
        return True
    except aiohttp.ClientConnectionError as e:
        print(f"Client connection error occurred: {e}")
        return True
    except aiohttp.ClientPayloadError as e:
        print(f"Client payload error occurred: {e}")
        return True
    except aiohttp.ClientError as e:
        print(f"An error occurred during the POST request: {e}")
        return True

async def update_sentiment_alerted_at(collection: AsyncIOMotorCollection, tracking_id: str):
    # Get the current timestamp
    current_time = datetime.now(timezone.utc)

    # Update the document with the new sentiment_alerted_at field
    result = await collection.update_one(
        {"tracking_id": tracking_id},
        {"$set": {"sentiment_alerted_at": current_time}}
    )

    if result.modified_count == 1:
        print(f"Successfully updated sentiment_alerted_at for _id: {tracking_id}")
    else:
        print(f"No document found with tracking_id: {tracking_id}")

# Execute the Main function every day at 7:49:00 AM PST
#@group.task(trigger=At(tz="America/Los_Angeles", hour=9, minute=35, second=0))
#@group.task(trigger=Once()) # Trigger once for testing
async def sentiment_analysis():
    print("🚀 sentiment_analysis() task has started executing!")  # Debugging print

    aa_motor_client = motor.motor_asyncio.AsyncIOMotorClient(os.getenv("AA_MONGO_CONNECTION"))
    tech_ticket_collection = aa_motor_client.get_database('metrics-database').get_collection('tech_tickets')
    yesterday = datetime.combine(datetime.now().date() - timedelta(days=1), datetime.min.time())
    one_week_ago = datetime.combine(datetime.now().date() - timedelta(days=3), datetime.min.time())

    # Query to find open tickets that have not been alerted in the last week or have never been alerted
    query = {
        "hs_pipeline_stage": 88539431,
        "$or": [
            {"sentiment_alerted_at": {"$exists": False}},
            {"sentiment_alerted_at": {"$lt": one_week_ago}},
        ],
        "last_update": {"$gte": yesterday}
    }
    projection = {"_id": 0, "tracking_id": 1, "slack_url": 1}
    results = await tech_ticket_collection.find(query, projection).to_list(length=None)
    print(f"Found {len(results)} open tickets")

    # Sentiment Analysis
    for res in results:
        thread_id = res['slack_url'].split('/')[-1]
        discord_status, title = await check_discord_status(thread_id=thread_id, tech_ticket_collection=tech_ticket_collection, tracking_id=res['tracking_id'])
        await asyncio.sleep(1)
        if discord_status != "closed":
            transcript = await parse_discord_messages(res['tracking_id'])

            if transcript['length'] > 10 and transcript['status'] == "open":
                # Create a CoversationObject for the sentiment analysis
                conversation_obj = CoversationObject(
                    thread_id=res['tracking_id'],
                    title=transcript.get('customer_name', 'Unknown'),
                    messages=[]
                )
                
                result = await sentiment_agent.run(
                    f"""Carefully analyze the sentiment of this conversation:
                    <conversation>
                    {transcript['conversation']}
                    </conversation>
                    """,
                    deps=conversation_obj,
                    model_settings=settings)
                #print(result.output)
                if result.output.escalation_required:
                    # Build JSON
                    webhook_data = {
                        #"content": f"Customer {transcript['customer_name']} is expressing severe dissatisfaction with their [tech support ticket]({transcript['thread_url']}).",
                        "embeds": [
                            {
                                "title": "Summary",
                                "description": result.output.technical_summary,
                                "color": 5814783,
                                "fields": [
                                    {
                                        "name": "🔥 Escalation Reason(s)",
                                        "value": "\n".join([f"• {reason}" for reason in result.output.primary_reasons]),
                                    },
                                    {
                                        "name": "📈 Frustration Pattern",
                                        "value": result.output.frustration_pattern,
                                        "inline": True
                                    },
                                    {
                                        "name": "💯 Confidence Score",
                                        "value": f"{round(result.output.confidence*100,2)}%",
                                        "inline": True
                                    },
                                    {
                                        "name": "❤️ Interaction Health",
                                        "value": str(result.output.interaction_health),
                                        "inline": True
                                    },
                                    {
                                        "name": "🗯️ Trigger Phrases",
                                        "value": "\n".join([f"• {phrase}" for phrase in result.output.trigger_phrases]),
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
                                        "value": result.output.recommended_action,
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

                    await post_msg(AUTOMATED_ESCALATIONS_CHANNEL, webhook_data)
                    await update_sentiment_alerted_at(tech_ticket_collection, res['tracking_id'])
            # else:
            #     print(f"Conversation length: {transcript['length']} too short. Not processed")

# Executes check_latency every hour
#@group.task(trigger=Every(minutes=60))
async def check_latency():
    print("⏳ Checking latency...")

    aa_motor_client = motor.motor_asyncio.AsyncIOMotorClient(os.getenv("AA_MONGO_CONNECTION"))
    tech_ticket_collection = aa_motor_client.get_database('metrics-database').get_collection('tech_tickets')
    employee_collection = aa_motor_client.get_database('metrics-database').get_collection('employees')

    # Query to find open tickets where avg > 600 or max > 1200 & notifications have not been sent
    query = {
        "$and": [
            { "latency_stats": { "$exists": True } },
            { "latency_stats.notified": { "$exists": False } },
            { "hs_pipeline_stage": 88539431 }, # 88539431 is In Progress, 88539432 is the closed stage
            {
                "$or": [
                    { "latency_stats.max_latency": { "$gt": 3750 } },
                    { "latency_stats.avg_latency": { "$gt": 1200 } }
                ]
            }
        ]
    }
    projection = {"_id": 0,
                  "tracking_id": 1, 
                  "hs_pipeline_stage": 1,
                  "technician" : 1,
                  "latency_stats": 1,
                  "slack_url": 1,
                  "open_date": 1}
    results = await tech_ticket_collection.find(query, projection).to_list(length=None)
    print(f"Found {len(results)} open tickets")
    employee_projection = {"discord_id":1, "name":1}

    # Latency Notification
    for res in results:
        # Skip tickets where min, avg, and max latency are the same
        if (res['latency_stats']['min_latency'] == res['latency_stats']['avg_latency'] == 
            res['latency_stats']['max_latency']):
            print(f"Skipping ticket {res['tracking_id']} - min/avg/max latency are all the same")
            continue
        
        # get thread id from discord url (slack_url)
        thread_id = res['slack_url'].split('/')[-1]
        discord_status, title = await check_discord_status(thread_id=thread_id, tech_ticket_collection=tech_ticket_collection, tracking_id=res['tracking_id'])
        
        if discord_status != "closed" and (res['latency_stats']['max_latency'] > 3250 or (res['latency_stats']['avg_latency'] > 1200 and res['latency_stats']['response_count'] > 10)):
            employee_query = { "hs_id": res['technician'] }
            employee_results = await employee_collection.find_one(employee_query, employee_projection)
            # tech_discord_id = employeeResults['discord_id'][0]  # Not sure if needed
            if employee_results is None:
                tech_name = "Unknown Employee"
            else:
                tech_name = employee_results['name']
            latency_webhook_data = {
                "embeds": [
                    {
                        "title": "🚨 Latency Notification: Delayed Support Response 🚨",
                        "description": "A support ticket has experienced severe response latency. Please review and take action as needed.",
                        "color": 16711680,
                        "fields": [
                            {
                                "name": "🗒️ Ticket Title",
                                "value": f"{title}",
                                "inline": False
                            },
                            {
                                "name": "📌 Ticket Link",
                                "value": f"[Click here to view the ticket]({res['slack_url']})",
                                "inline": False
                            },
                            {
                                "name": "👤 Ticket Owner",
                                "value": str(tech_name),  
                                "inline": True
                            },
                            {
                                "name": "⏳ Min Latency",
                                "value": f"{round(res['latency_stats']['min_latency'],2)} minutes", 
                                "inline": True
                            },
                            {
                                "name": "⏳ Avg Latency",
                                "value": f"{round(res['latency_stats']['avg_latency'],2)} minutes",  
                                "inline": True
                            },
                            {
                                "name": "⏳ Max Latency",
                                "value": f"{round(res['latency_stats']['max_latency'],2)} minutes",  
                                "inline": True
                            }
                        ],
                        "footer": {
                            "text": "Support Team • Please prioritize this ticket. Opened "
                        },
                        "timestamp": res['open_date'].isoformat() if hasattr(res['open_date'], 'isoformat') else res['open_date']
                    }
                ]
            }

            notification_successful = await post_msg(LATENCY_WARNINGS_CHANNEL, latency_webhook_data)
            print("notification sent")
            now = datetime.now()  # Use UTC for consistency
            if notification_successful:
                query = { "tracking_id": res['tracking_id'] } 
                update = { "$set": { "latency_stats.notified": now }} 
                await tech_ticket_collection.update_one(query, update)

async def check_discord_status(thread_id: str, tech_ticket_collection: motor.motor_asyncio.AsyncIOMotorCollection, tracking_id: str, hs_pipeline_stage=88539431):
    get_thread_tags_url = f"http://134.195.91.7:5005/discord/{thread_id}/get_tags"

    # Create the authentication string
    auth_string = f'{USERNAME}:{PASSWORD}'
    auth_bytes = auth_string.encode('utf-8')
    base64_bytes = base64.b64encode(auth_bytes)
    base64_string = base64_bytes.decode('utf-8')

    headers = {
        'Authorization': f'Basic {base64_string}'
    }

    if hs_pipeline_stage == 88539432: # Ticket is already closed, just return closed
        return "closed", "Ticket Closed"
    else:
        try:
            await asyncio.sleep(0.5)
            async with aiohttp.ClientSession() as session:
                async with session.get(get_thread_tags_url, headers=headers) as response:
                    response_data = await response.json()
                    if response.status == 200:
                        #print("POST request successful!")
                        if response_data.get('status') == 'closed':
                            # ticket is closed, we need to update db to reflect this
                            print("Ticket Closed on Discord - Updating db to reflect this")
                            results = await tech_ticket_collection.update_one({"tracking_id": tracking_id}, {"$set": {"hs_pipeline_stage": 88539432}})
                        return (response_data.get('status'), response_data.get('title','Unknown'))
                    else:
                        print(f"POST request failed with status code: {response.status}")
                        return ("unknown", "unknown")
        except aiohttp.ClientResponseError as e:
            print(f"Client response error occurred: {e}")
            return ("unknown", "unknown")
        except aiohttp.ClientConnectionError as e:
            print(f"Client connection error occurred: {e}")
            return ("unknown", "unknown")
        except aiohttp.ClientPayloadError as e:
            print(f"Client payload error occurred: {e}")
            return ("unknown", "unknown")
        except aiohttp.ClientError as e:
            print(f"An error occurred during the POST request: {e}")
            return ("unknown", "unknown")


# Removed aioclock scheduling - now runs immediately
if __name__ == "__main__":
    async def run_all():
        print("🚀 Starting sentiment analysis and latency checks...")
        await sentiment_analysis()
        await check_latency()
        print("✅ All tasks completed!")
    
    asyncio.run(run_all())
