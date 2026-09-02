"""Add durable worker lease ownership to runs."""

import sqlalchemy as sa
from alembic import op

revision = "0002_run_leases"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("runs", sa.Column("lease_owner", sa.String(length=96), nullable=True))
    op.add_column("runs", sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("runs", sa.Column("attempt", sa.Integer(), server_default="0", nullable=False))
    op.create_index(op.f("ix_runs_lease_owner"), "runs", ["lease_owner"], unique=False)
    op.create_index(op.f("ix_runs_lease_expires_at"), "runs", ["lease_expires_at"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_runs_lease_expires_at"), table_name="runs")
    op.drop_index(op.f("ix_runs_lease_owner"), table_name="runs")
    op.drop_column("runs", "attempt")
    op.drop_column("runs", "lease_expires_at")
    op.drop_column("runs", "lease_owner")
