"""Cross-worker search-provider health (shared circuit).

Stores the account-level outage (auth/quota) of a search provider so one worker
that discovers it stops all workers from hammering a dead key — without stopping
any process.
"""

import sqlalchemy as sa
from alembic import op

revision = "0008_provider_health"
down_revision = "0007_fix_null_resume_payload"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "provider_health",
        sa.Column("provider", sa.String(length=32), nullable=False),
        sa.Column("reason", sa.String(length=32), nullable=False),
        sa.Column("open_until", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("provider"),
    )


def downgrade() -> None:
    op.drop_table("provider_health")
