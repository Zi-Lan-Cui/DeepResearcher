"""Add shared login throttling state."""

import sqlalchemy as sa
from alembic import op

revision = "0006_login_throttles"
down_revision = "0005_tool_cache"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "login_throttles",
        sa.Column("key_hash", sa.String(length=64), nullable=False),
        sa.Column("attempt_count", sa.Integer(), nullable=False),
        sa.Column("window_started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("blocked_until", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("key_hash"),
    )
    op.create_index(
        op.f("ix_login_throttles_blocked_until"),
        "login_throttles",
        ["blocked_until"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_login_throttles_blocked_until"), table_name="login_throttles")
    op.drop_table("login_throttles")
