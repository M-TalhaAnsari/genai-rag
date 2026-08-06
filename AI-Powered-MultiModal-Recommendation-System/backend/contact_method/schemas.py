from pydantic import BaseModel, EmailStr


class EmailComposeRequest(BaseModel):
    email: EmailStr
    message: str
    subject: str = "Reservation Request"


class EmailComposeResponse(BaseModel):
    email: str
    gmail_url: str

class WhatsAppRequest(BaseModel):
    message: str