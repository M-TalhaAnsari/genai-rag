"""add email_verified to users

Revision ID: 0003
Revises: 0002
"""
from alembic import op
import sqlalchemy as sa

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "users",
        sa.Column("email_verified", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    # Google users created before this migration are trustworthy —
    # Google already verified their email. Backfill so they aren't
    # locked out by the new login check.
    op.execute("UPDATE users SET email_verified = TRUE WHERE google_id IS NOT NULL")


def downgrade():
    op.drop_column("users", "email_verified")