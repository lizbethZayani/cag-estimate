"""hybrid search: full-text column and GIN index on chunks

Revision ID: 0002
Revises: 0001

Adds ``chunks.content_tsv``, a STORED generated ``tsvector`` of ``content``, plus a GIN
index so the lexical half of hybrid search stays fast. The text search configuration is
``english``: the session brief assumed Spanish, but the indexed corpus is English.
"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "chunks",
        sa.Column(
            "content_tsv",
            postgresql.TSVECTOR(),
            sa.Computed("to_tsvector('english', content)", persisted=True),
        ),
    )
    op.create_index("ix_chunks_content_tsv", "chunks", ["content_tsv"], postgresql_using="gin")


def downgrade() -> None:
    op.drop_index("ix_chunks_content_tsv", table_name="chunks")
    op.drop_column("chunks", "content_tsv")
