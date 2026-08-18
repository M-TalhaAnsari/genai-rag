"""add google oauth fields to users

Revision ID: 0002
Revises: 0001
Create Date: 2026-08-15
"""
from alembic import op
import sqlalchemy as sa

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade():
    op.alter_column("users", "hashed_password", nullable=True)
    op.add_column(
        "users",
        sa.Column("auth_provider", sa.String(), nullable=False, server_default="local"),
    )
    op.add_column("users", sa.Column("google_id", sa.String(), nullable=True))
    op.create_index("ix_users_google_id", "users", ["google_id"], unique=True)


def downgrade():
    op.drop_index("ix_users_google_id", table_name="users")
    op.drop_column("users", "google_id")
    op.drop_column("users", "auth_provider")
    op.alter_column("users", "hashed_password", nullable=False)