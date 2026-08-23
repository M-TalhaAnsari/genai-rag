"""
backend/models/db_models.py
-----------------------------
All SQLAlchemy ORM models (database tables).
"""

from sqlalchemy import (
    Column, Integer, String, Boolean, Float,
    DateTime, Text, ForeignKey
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from backend.core.database import Base


import enum
import uuid
from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Enum, ForeignKey, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship


class Restaurant(Base):
    __tablename__ = "restaurants"

    id                   = Column(Integer, primary_key=True, index=True)

    # Core identity
    name                 = Column(String, nullable=False)
    cuisine              = Column(String, nullable=False)
    all_cuisines         = Column(Text, nullable=True)       # JSON list
    city                 = Column(String, nullable=False)
    area                 = Column(String, nullable=True)     # neighbourhood/sector
    postal_code          = Column(String, nullable=True)

    # Contact
    address              = Column(String, nullable=True)
    phone                = Column(String, nullable=True)
    email                = Column(String, nullable=True)
    website              = Column(String, nullable=True)
    menu_url             = Column(String, nullable=True)

    # Location
    latitude             = Column(Float, nullable=True)
    longitude            = Column(Float, nullable=True)

    # Quality signals
    rating               = Column(Float, nullable=True)
    review_count         = Column(Integer, nullable=True)
    price_level          = Column(String, nullable=True)
    reviews_distribution = Column(Text, nullable=True)       # JSON

    # Rich content
    description          = Column(Text, nullable=True)
    opening_hours        = Column(Text, nullable=True)       # JSON
    photos               = Column(Text, nullable=True)       # JSON list of URLs
    tags                 = Column(Text, nullable=True)       # JSON list

    # Provenance
    source               = Column(String, nullable=True)     # apify | osm | foursquare
    external_id          = Column(String, nullable=True)     # Google placeId

    # Vector index flag
    is_embedded          = Column(Boolean, default=False, nullable=False)

    # Timestamps
    created_at           = Column(DateTime, server_default=func.now())
    updated_at           = Column(DateTime, server_default=func.now(), onupdate=func.now())

    # Relationships
    reviews              = relationship("Review", back_populates="restaurant",
                                        cascade="all, delete-orphan")


class Review(Base):
    __tablename__ = "reviews"

    id              = Column(Integer, primary_key=True, index=True)
    restaurant_id   = Column(Integer, ForeignKey("restaurants.id"),
                             nullable=False, index=True)
    reviewer_name   = Column(String, nullable=True)
    rating          = Column(Float, nullable=True)
    text            = Column(Text, nullable=True)
    published_date  = Column(String, nullable=True)
    source          = Column(String, nullable=True)
    created_at      = Column(DateTime, server_default=func.now())

    restaurant      = relationship("Restaurant", back_populates="reviews")


class UserFeedback(Base):
    __tablename__ = "user_feedback"

    id              = Column(Integer, primary_key=True, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    restaurant_id   = Column(Integer, nullable=False, index=True)
    restaurant_name = Column(String, nullable=False)
    cuisine         = Column(String, nullable=False)
    city            = Column(String, nullable=False)
    signal          = Column(Integer, nullable=False)    # 1=like, -1=dislike
    query           = Column(String, nullable=True)
    created_at      = Column(DateTime, server_default=func.now())


class UserProfile(Base):
    __tablename__ = "user_profiles"

    id                  = Column(Integer, primary_key=True, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    preferred_cuisines  = Column(Text, default="[]")
    avoided_cuisines    = Column(Text, default="[]")
    preferred_cities    = Column(Text, default="[]")
    preference_vector   = Column(Text, default="[]")
    feedback_count      = Column(Integer, default=0)
    updated_at          = Column(DateTime, server_default=func.now(),
                                 onupdate=func.now())


class SearchLog(Base):
    __tablename__ = "search_logs"

    id           = Column(Integer, primary_key=True, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    query        = Column(String, nullable=False)
    result_count = Column(Integer, default=0)
    created_at   = Column(DateTime, server_default=func.now())


class ConversationHistory(Base):
    __tablename__ = "conversation_history"

    id         = Column(Integer, primary_key=True, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    role       = Column(String, nullable=False)
    content    = Column(Text, nullable=False)
    query      = Column(String, nullable=True)
    created_at = Column(DateTime, server_default=func.now())


class UserMemorySummary(Base):
    __tablename__ = "user_memory_summaries"

    id         = Column(Integer, primary_key=True, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    summary    = Column(Text, nullable=False)
    turn_count = Column(Integer, default=0)
    updated_at = Column(DateTime, server_default=func.now(),
                        onupdate=func.now())


"""
Additions to backend/models/db_models.py

Add the User and RefreshToken tables below. Then change the four
existing tables that currently take a free-text user_id string —
UserFeedback, UserProfile, ConversationHistory, UserMemorySummary —
to a real foreign key. This is the change that actually closes the
"anyone can pass anyone else's user_id" hole; adding auth without this
FK change means auth exists but nothing enforces it downstream.
"""



class UserRole(str, enum.Enum):
    user = "user"
    admin = "admin"


class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    auth_provider = Column(String, default="local", nullable=False)  # "local" | "google" | "google_and_local"
    google_id = Column(String, unique=True, nullable=True, index=True)
    role = Column(Enum(UserRole), default=UserRole.user, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    email_verified = Column(Boolean, nullable=False, server_default="false")
    refresh_tokens = relationship(
        "RefreshToken", back_populates="user", cascade="all, delete-orphan"
    )


class RefreshToken(Base):
    __tablename__ = "refresh_tokens"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    jti = Column(UUID(as_uuid=True), unique=True, nullable=False, index=True)
    expires_at = Column(DateTime, nullable=False)
    revoked = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="refresh_tokens")

