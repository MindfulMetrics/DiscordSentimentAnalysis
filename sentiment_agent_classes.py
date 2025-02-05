from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from dataclasses import dataclass
from datetime import date, datetime


@dataclass
class MessageObject:
    id: str
    author: str
    content: str
    is_customer: bool
    author_id: str
    edited_at: datetime | None = None


@dataclass
class CoversationObject:
    thread_id: str
    title: str
    messages: list[MessageObject]


class SummarizedOutput(BaseModel):
    """
    A schema for real-time sentiment analysis of troubleshooting forum conversations.
    The analysis focuses on the troubleshooting process and monitors for severe frustration 
    directed at the technician, which may require escalation.
    """
    wholistic_reasoning: str = Field(
        ...,
        description="A concise explanation for the overall sentiment score, focusing on key factors such as customer frustration, troubleshooting difficulty, and sentiment toward the technician."
    )
    wholistic_score: float = Field(
        50,
        ge=0, le=100,
        description="The overall sentiment score for the conversation (0 = very negative, 50 = neutral, 100 = very positive)."
    )
    final_summary: str = Field(
        ...,
        description="A brief explanation for the sentiment score of the final message in the conversation."
    )
    final_score: float = Field(
        50,
        ge=0, le=100,
        description="The sentiment score of the most recent message (0 = very negative, 50 = neutral, 100 = very positive)."
    )
    needs_escalation: bool = Field(
        False,
        description="A flag indicating whether the conversation shows severe frustration or major dissatisfaction directed at the technician, warranting escalation to management."
    )
