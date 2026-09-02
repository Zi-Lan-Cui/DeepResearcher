"""Add detailed and aggregated run usage accounting."""

import sqlalchemy as sa
from alembic import op

revision = "0004_run_usage"
down_revision = "0003_run_control_event_seq"
branch_labels = None
depends_on = None


def upgrade() -> None:
    for name in (
        "llm_call_count",
        "input_tokens",
        "output_tokens",
        "cached_input_tokens",
        "external_request_count",
        "peak_llm_concurrency",
    ):
        op.add_column("runs", sa.Column(name, sa.Integer(), server_default="0", nullable=False))
    op.add_column(
        "runs",
        sa.Column("estimated_cost_usd", sa.Numeric(14, 8), server_default="0", nullable=False),
    )
    op.create_table(
        "run_usage",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("run_id", sa.String(length=64), nullable=False),
        sa.Column("category", sa.String(length=32), nullable=False),
        sa.Column("component", sa.String(length=64), nullable=False),
        sa.Column("provider", sa.String(length=64), nullable=True),
        sa.Column("model", sa.String(length=128), nullable=True),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("usage_estimated", sa.Boolean(), nullable=False),
        sa.Column("input_tokens", sa.Integer(), nullable=False),
        sa.Column("output_tokens", sa.Integer(), nullable=False),
        sa.Column("cached_input_tokens", sa.Integer(), nullable=False),
        sa.Column("cost_usd", sa.Numeric(14, 8), nullable=False),
        sa.Column("duration_ms", sa.Integer(), nullable=False),
        sa.Column("detail_json", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_run_usage_run_id"), "run_usage", ["run_id"], unique=False)
    op.create_index(op.f("ix_run_usage_category"), "run_usage", ["category"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_run_usage_category"), table_name="run_usage")
    op.drop_index(op.f("ix_run_usage_run_id"), table_name="run_usage")
    op.drop_table("run_usage")
    op.drop_column("runs", "estimated_cost_usd")
    for name in reversed(
        (
            "llm_call_count",
            "input_tokens",
            "output_tokens",
            "cached_input_tokens",
            "external_request_count",
            "peak_llm_concurrency",
        )
    ):
        op.drop_column("runs", name)
