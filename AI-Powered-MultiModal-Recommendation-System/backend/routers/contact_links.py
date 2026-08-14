"""
backend/routers/contact_links.py
----------------------------------
Contact links router.

GET /restaurants/{id}/contact-links
  Returns clickable email, WhatsApp, website, and menu links for a
  restaurant card. Only channels with actual data are returned.
  The user clicks a link — their own app handles the rest.
  No webhooks. No LLM. No n8n.

Frontend behaviour per channel:
  email      → mailto: href → opens default email client, message pre-filled
  whatsapp   → wa.me link  → opens WhatsApp web/app, message pre-filled
  website    → plain URL   → opens in new tab
  menu       → plain URL   → opens in new tab
"""

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from backend.core.database import get_db
from backend.models.db_models import Restaurant
from backend.models.schemas import ContactLinksResponse
from backend.services.contact_service import get_contact_links


router = APIRouter(
    prefix="/restaurants",
    tags=["contact links"]
)


@router.get(
    "/{restaurant_id}/contact-links",
    response_model=ContactLinksResponse
)
async def contact_links(
    restaurant_id: int,
    user_name: str = Query(default="Guest",
                           description="User's name — pre-filled in the message"),
    user_query: str = Query(default="",
                            description="What the user was searching for — context for the message"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get all available clickable contact options for a restaurant.

    Call this when a user clicks "Contact" on a restaurant card.
    Returns only the channels that have real data — never shows
    an email button if there's no email address in the database.

    Response fields:
      email_href     → mailto: URL — open in user's email client
      whatsapp_href  → wa.me URL  — open WhatsApp with pre-filled message
      website        → restaurant website URL
      menu_url       → direct menu link if available
      available_channels → ["email", "whatsapp", "website", "menu"]
                           (only populated ones — use this to decide
                            which buttons to render in the frontend)

    Example response when a restaurant has phone + website but no email:
    {
      "restaurant_id": 42,
      "restaurant_name": "El Momento",
      "email": null,
      "email_href": null,
      "phone": "+92 311 1100317",
      "whatsapp_href": "https://wa.me/923111100317?text=Hi+El+Momento...",
      "website": "https://elmomento.pk/el-momento-islamabad/",
      "menu_url": "https://elmomento.pk/menu-islamabad/",
      "available_channels": ["whatsapp", "website", "menu"]
    }
    """
    result = await db.execute(
        select(Restaurant).where(Restaurant.id == restaurant_id)
    )
    restaurant = result.scalars().first()

    if not restaurant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Restaurant {restaurant_id} not found."
        )

    links = get_contact_links(
        restaurant_id=restaurant.id,
        restaurant_name=restaurant.name,
        email=restaurant.email,
        phone=restaurant.phone,
        website=restaurant.website,
        menu_url=restaurant.menu_url,
        user_name=user_name,
        user_query=user_query,
    )

    return ContactLinksResponse(**links)