"""Add persistent tool cache and cache-savings aggregates."""

import sqlalchemy as sa
from alembic import op

revision = "0005_tool_cache"
down_revision = "0004_run_usage"
branch_labels = None
depends_on = None


def upgrade() -> None:
    for name in (
        "cache_hit_count",
        "saved_external_request_count",
        "saved_llm_call_count",
        "saved_tokens",
    ):
        op.add_column("runs", sa.Column(name, sa.Integer(), server_default="0", nullable=False))
    op.add_column(
        "runs",
        sa.Column("saved_cost_usd", sa.Numeric(14, 8), server_default="0", nullable=False),
    )
    op.create_table(
        "tool_cache_entries",
        sa.Column("namespace", sa.String(length=32), nullable=False),
        sa.Column("cache_key", sa.String(length=64), nullable=False),
        sa.Column("value_json", sa.JSON(), nullable=False),
        sa.Column("metrics_json", sa.JSON(), nullable=True),
        sa.Column("content_hash", sa.String(length=64), nullable=True),
        sa.Column("schema_version", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_accessed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("hit_count", sa.Integer(), server_default="0", nullable=False),
        sa.PrimaryKeyConstraint("namespace", "cache_key"),
    )
    op.create_index(
        op.f("ix_tool_cache_entries_content_hash"),
        "tool_cache_entries",
        ["content_hash"],
        unique=False,
    )
    op.create_index(
        op.f("ix_tool_cache_entries_expires_at"),
        "tool_cache_entries",
        ["expires_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_tool_cache_entries_expires_at"), table_name="tool_cache_entries")
    op.drop_index(op.f("ix_tool_cache_entries_content_hash"), table_name="tool_cache_entries")
    op.drop_table("tool_cache_entries")
    op.drop_column("runs", "saved_cost_usd")
    for name in reversed(
        (
            "cache_hit_count",
            "saved_external_request_count",
            "saved_llm_call_count",
            "saved_tokens",
        )
    ):
        op.drop_column("runs", name)
