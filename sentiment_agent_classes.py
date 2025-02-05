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
    wholistic_reasoning: str = Field(
        ..., description="A short and concise reasoning behind the wholistic_score grade")
    wholistic_score: float = Field(
        50, description="a sentiment score of the entire conversation")
    final_summary: str = Field(
        ..., description="A short and concise reasoning behind the final_score grade")
    final_score: float = Field(
        50, description="a sentiment score of the last message of the conversation")
