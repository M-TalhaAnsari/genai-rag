"""
backend/models.py
------------------
SQLAlchemy models for the full production schema.

Tables:
  Restaurant    — full restaurant profile (name, contact, rating, etc.)
  Review        — individual user reviews (one-to-many with Restaurant)
  UserFeedback  — thumbs up/down signals from users
  UserProfile   — derived preference profile per user
  SearchLog     — every /search and /recommend call logged for analytics
"""

from sqlalchemy import (
    Column, Integer, String, Boolean, Float,
    DateTime, Text, ForeignKey
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from backend.db.database import Base


class Restaurant(Base):

    __tablename__ = "restaurants"

    id            = Column(Integer, primary_key=True, index=True)

    # Core identity
    name          = Column(String, nullable=False)
    cuisine       = Column(String, nullable=False)
    city          = Column(String, nullable=False)

    # Contact
    address       = Column(String, nullable=True)
    phone         = Column(String, nullable=True)
    website       = Column(String, nullable=True)
    menu_url      = Column(String, nullable=True)

    # Location
    latitude      = Column(Float, nullable=True)
    longitude     = Column(Float, nullable=True)

    # Quality signals
    rating        = Column(Float, nullable=True)       # 0.0 – 5.0
    review_count  = Column(Integer, nullable=True)
    price_level   = Column(String, nullable=True)      # "$" | "$$" | "$$$" | "$$$$"

    # Rich content (stored as JSON strings)
    description   = Column(Text, nullable=True)
    opening_hours = Column(Text, nullable=True)        # JSON: {"Monday": "9am-10pm", ...}
    photos        = Column(Text, nullable=True)        # JSON: ["url1", "url2", ...]
    tags          = Column(Text, nullable=True)        # JSON: ["rooftop", "family-friendly"]

    # Provenance
    source        = Column(String, nullable=True)      # "apify" | "osm" | "foursquare"
    external_id   = Column(String, nullable=True)      # original ID from source API

    # Vector index flag
    is_embedded   = Column(Boolean, default=False, nullable=False)

    # Timestamps
    created_at    = Column(DateTime, server_default=func.now())
    updated_at    = Column(DateTime, server_default=func.now(), onupdate=func.now())

    # Relationship
    reviews       = relationship("Review", back_populates="restaurant",
                                 cascade="all, delete-orphan")


class Review(Base):

    __tablename__ = "reviews"

    id              = Column(Integer, primary_key=True, index=True)
    restaurant_id   = Column(Integer, ForeignKey("restaurants.id"), nullable=False, index=True)

    reviewer_name   = Column(String, nullable=True)
    rating          = Column(Float, nullable=True)      # 1.0 – 5.0
    text            = Column(Text, nullable=True)
    published_date  = Column(String, nullable=True)     # stored as string from source
    source          = Column(String, nullable=True)     # "google" | "foursquare" etc.

    created_at      = Column(DateTime, server_default=func.now())

    restaurant      = relationship("Restaurant", back_populates="reviews")


class UserFeedback(Base):

    __tablename__ = "user_feedback"

    id              = Column(Integer, primary_key=True, index=True)
    user_id         = Column(String, nullable=False, index=True)
    restaurant_id   = Column(Integer, nullable=False, index=True)
    restaurant_name = Column(String, nullable=False)
    cuisine         = Column(String, nullable=False)
    city            = Column(String, nullable=False)
    signal          = Column(Integer, nullable=False)   # 1 = like, -1 = dislike
    query           = Column(String, nullable=True)
    created_at      = Column(DateTime, server_default=func.now())


class UserProfile(Base):

    __tablename__ = "user_profiles"

    id                  = Column(Integer, primary_key=True, index=True)
    user_id             = Column(String, unique=True, nullable=False, index=True)
    preferred_cuisines  = Column(Text, default="[]")
    avoided_cuisines    = Column(Text, default="[]")
    preferred_cities    = Column(Text, default="[]")
    preference_vector   = Column(Text, default="[]")
    feedback_count      = Column(Integer, default=0)
    updated_at          = Column(DateTime, server_default=func.now(), onupdate=func.now())


class SearchLog(Base):

    __tablename__ = "search_logs"

    id           = Column(Integer, primary_key=True, index=True)
    user_id      = Column(String, nullable=True)
    query        = Column(String, nullable=False)
    result_count = Column(Integer, default=0)
    created_at   = Column(DateTime, server_default=func.now())
