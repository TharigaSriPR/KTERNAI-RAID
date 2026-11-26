"""Ingestion module for project signals (Emails, Chats, MOM/Meetings, Project Plans, Defects)"""
from .base_ingestor import BaseIngestor
from .email_ingestor import EmailIngestor
from .chat_ingestor import ChatIngestor
from .meeting_ingestor import MeetingIngestor
from .plan_ingestor import PlanIngestor
from .defect_ingestor import DefectIngestor

__all__ = [
    "BaseIngestor",
    "EmailIngestor",
    "ChatIngestor",
    "MeetingIngestor",
    "PlanIngestor",
    "DefectIngestor"
]
