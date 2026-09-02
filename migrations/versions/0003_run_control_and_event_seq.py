"""Persist cancellation, resume payload and event sequence."""

import sqlalchemy as sa
from alembic import op

revision = "0003_run_control_event_seq"
down_revision = "0002_run_leases"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "runs", sa.Column("cancellation_requested_at", sa.DateTime(timezone=True), nullable=True)
    )
    op.add_column("runs", sa.Column("resume_payload", sa.JSON(), nullable=True))
    op.add_column("runs", sa.Column("event_seq", sa.Integer(), server_default="0", nullable=False))
    op.execute(
        "UPDATE runs SET event_seq = COALESCE("
        "(SELECT MAX(run_events.seq) FROM run_events WHERE run_events.run_id = runs.id), 0)"
    )


def downgrade() -> None:
    op.drop_column("runs", "event_seq")
    op.drop_column("runs", "resume_payload")
    op.drop_column("runs", "cancellation_requested_at")
