"""Normalize JSON-null resume_payload to SQL NULL.

The resume single-consumption CAS matches on `resume_payload IS NULL`, but the
JSON column stored Python `None` as the JSON literal `null` (none_as_null is
off by default), so real clarifications could never be resumed. The model now
serializes None as SQL NULL; this backfill repairs rows already stuck as JSON
null so already-parked runs can be answered again.
"""

import sqlalchemy as sa
from alembic import op

revision = "0007_fix_null_resume_payload"
down_revision = "0006_login_throttles"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute(
            "UPDATE runs SET resume_payload = NULL "
            "WHERE resume_payload IS NOT NULL AND trim(resume_payload::text) = 'null'"
        )
    else:  # SQLite stores the JSON column as text
        op.execute(
            "UPDATE runs SET resume_payload = NULL "
            "WHERE resume_payload IS NOT NULL AND trim(resume_payload) = 'null'"
        )


def downgrade() -> None:
    # Readers treat JSON null and SQL NULL identically; nothing reversible to restore.
    pass
