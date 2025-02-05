from pydantic import BaseModel, Field
from typing import List
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


class EscalationAnalysis(BaseModel):
    """
    A schema for real-time sentiment analysis of troubleshooting forum conversations.
    The analysis focuses on the troubleshooting process and monitors for severe frustration 
    directed at the technician, which may require escalation.
    """
    escalation_required: bool = Field(...,
                                      description="True if immediate management escalation needed")

    confidence: float = Field(...,
                              description="Confidence score 0-1 for escalation decision",
                              ge=0, le=1)

    primary_reasons: List[str] = Field(...,
        description="Key escalation factors from this analysis.",
        examples=[["Customer threatened legal action", "3+ frustrated messages"]]
    )

    # sentiment_timeline: List[float] = Field(...,
    #     description="Sentiment scores (0-100) for each customer message in order"
    # )

    technical_summary: str = Field(...,
        description="Brief and concise summary of the technical issue customer is facing, along with steps taken to try and resolve it."
    )

    interaction_health: float = Field(...,
        description="Quality of communication (0=broken, 100=excellent)",
        ge=0, le=100
    )

    frustration_pattern: str = Field(...,
                                     enum=["isolated", "persistent",
                                           "escalating", "de-escalating"],
                                     description="Pattern of customer frustration")

    trigger_phrases: List[str] = Field(...,
                                       description="Exact phrases requiring escalation",
                                       examples=[["I'm losing money", "This is unacceptable"]])

    recommended_action: str = Field(...,
                                    enum=["monitor", "escalate",
                                          "urgent_escalate"],
                                    description="Immediate response recommendation")
