"""
backend/services/contact_service.py
-------------------------------------
Generates clickable contact links for restaurant cards.

No webhooks. No n8n calls. No LLM.
Just clean URL generation — the user's own email client / WhatsApp
handles the actual sending after they click.

EMAIL
------
Generates a mailto: href with pre-filled subject and body.
Clicking it opens the user's default email client (Gmail, Outlook, etc.)
with everything filled in — user just presses Send.

WHATSAPP
---------
Generates a wa.me link with a pre-encoded message.
Clicking it opens WhatsApp (web or app) with the message pre-filled.
Phone number is normalised to E.164 format (required by wa.me).

WEBSITE / MENU
---------------
Just the stored URL — opens in a new tab.

MESSAGE GENERATION
-------------------
We generate a short, natural message using a template (no LLM needed
for this simple case — template quality is good enough and saves tokens).
If you later want LLM-generated messages, just swap the template function
for an LLM call — the service interface stays the same.
"""

import re
from urllib.parse import quote


# ── Phone normalisation ────────────────────────────────────────────────────

def normalise_phone(phone: str, default_country_code: str = "92") -> str | None:
    """
    Convert a Pakistani phone number to E.164 format for wa.me links.

    """
    if not phone:
        return None

    # Strip everything except digits and leading +
    digits = re.sub(r"[^\d+]", "", phone.strip())

    # Already E.164 with +
    if digits.startswith("+"):
        return digits[1:]

    # Starts with country code without +
    if digits.startswith("92") and len(digits) >= 11:
        return digits

    # Starts with 0 (local Pakistani format)
    if digits.startswith("0") and len(digits) >= 10:
        return default_country_code + digits[1:]

    # Bare number — prepend Pakistan country code
    if len(digits) >= 9:
        return default_country_code + digits

    return None


# ── Message template ───────────────────────────────────────────────────────

def build_contact_message(
    restaurant_name: str,
    user_name: str,
    user_query: str,
    channel: str = "whatsapp"
) -> str:
    """
    Build a short, natural pre-filled message.

    channel: "email" | "whatsapp"
    Email gets a slightly more formal tone.
    WhatsApp is kept very short — walls of text don't work on WhatsApp.
    """
    name = user_name.strip() if user_name else "a customer"
    query_note = f" I was looking for: {user_query.strip()}." if user_query else ""

    if channel == "whatsapp":
        return (
            f"Hi {restaurant_name}! 👋 I'm {name}.{query_note} "
            f"Could you please share more details? Thank you!"
        )
    else:  # email
        return (
            f"Dear {restaurant_name} Team,\n\n"
            f"My name is {name} and I am interested in visiting your restaurant.{query_note}\n\n"
            f"Could you please provide more information?\n\n"
            f"Best regards,\n{name}"
        )


# ── Link builders ───────────────────────────────────────────────────────────

def build_email_href(
    email: str,
    restaurant_name: str,
    user_name: str,
    user_query: str
) -> str:
    """
    Build a mailto: href with pre-filled subject and body.
    Clicking this opens the user's email client with everything ready.
    """
    subject = quote(f"Inquiry about {restaurant_name}")
    body    = quote(build_contact_message(
        restaurant_name, user_name, user_query, channel="email"
    ))
    return f"mailto:{email}?subject={subject}&body={body}"


def build_whatsapp_href(
    phone: str,
    restaurant_name: str,
    user_name: str,
    user_query: str
) -> str | None:
    """
    Build a wa.me link with a pre-filled message.
    Returns None if the phone number can't be normalised.
    """
    normalised = normalise_phone(phone)
    if not normalised:
        return None

    message = build_contact_message(
        restaurant_name, user_name, user_query, channel="whatsapp"
    )
    return f"https://wa.me/{normalised}?text={quote(message)}"


# ── Main service function ──────────────────────────────────────────────────

def get_contact_links(
    restaurant_id: int,
    restaurant_name: str,
    email: str | None,
    phone: str | None,
    website: str | None,
    menu_url: str | None,
    user_name: str = "Guest",
    user_query: str = ""
) -> dict:
    """
    Generate all available clickable contact links for a restaurant.

    Returns a dict matching ContactLinksResponse schema.
    Only channels that have actual data are populated.
    """
    available_channels = []
    result = {
        "restaurant_id":   restaurant_id,
        "restaurant_name": restaurant_name,
        "email":           None,
        "email_href":      None,
        "phone":           None,
        "whatsapp_href":   None,
        "website":         None,
        "menu_url":        None,
        "available_channels": [],
    }

    if email and "@" in email:
        result["email"]      = email
        result["email_href"] = build_email_href(
            email, restaurant_name, user_name, user_query
        )
        available_channels.append("email")

    if phone:
        wa_href = build_whatsapp_href(
            phone, restaurant_name, user_name, user_query
        )
        if wa_href:
            result["phone"]          = phone
            result["whatsapp_href"]  = wa_href
            available_channels.append("whatsapp")

    if website:
        result["website"] = website
        available_channels.append("website")

    if menu_url:
        result["menu_url"] = menu_url
        available_channels.append("menu")

    result["available_channels"] = available_channels
    return result