"""Canonical relational schema for bot memory (SQLAlchemy Core).

Domain code builds Core ``select`` / ``insert`` / ``update`` / ``delete`` against
``memory_table``. The store and providers compile those clauses to dialect SQL.

The episodic graph lives beside it: ``node_table`` (episodes, persons, concepts),
``person_ref_table`` (which person each participant ref belongs to; a ref is its primary key, so
each ref is exactly one person's), ``part_table`` (each node's ordered text and image parts),
``edge_table`` (typed links),
``vector_table`` (the canonical vector of a node in one embedding space), and
``index_attempt_table`` (failed indexing attempts, so backfill never retries what cannot
succeed in a space). No foreign key cascades are relied on: Memory deletes dependent rows
explicitly inside the transaction that forgets a node.

``identity_table`` keeps the identities an ``identification.Store`` minted or enrolled (voices
and faces), beside the memory they are referenced from as ``voice:<id>`` / ``face:<id>``
participant refs. Their prints live only in provider vector tables.

Provider-created vector tables (memory's ``bot_memory_vec_*`` index and the identity prints'
``bot_identity_vec_*``) are not part of this metadata and are never managed by migrations.
"""

from __future__ import annotations

from sqlalchemy import Column
from sqlalchemy import Float
from sqlalchemy import Index
from sqlalchemy import Integer
from sqlalchemy import LargeBinary
from sqlalchemy import MetaData
from sqlalchemy import PrimaryKeyConstraint
from sqlalchemy import Table
from sqlalchemy import Text
from sqlalchemy import UniqueConstraint
from sqlalchemy import func

# Logical table name (dialect-neutral identifier).
MEMORY_TABLE = "bot_memory"

# Logical table name for the short-term memory register (dialect-neutral identifier).
STM_MEMORY_TABLE = "bot_stm_memory"

# Logical table name for the short-term memory of turns: what the bot heard and said.
STM_TURN_TABLE = "bot_stm_turn"

# Episodic graph tables.
NODE_TABLE = "bot_memory_node"
PART_TABLE = "bot_memory_part"
EDGE_TABLE = "bot_memory_edge"
VECTOR_TABLE = "bot_memory_vector"
INDEX_ATTEMPT_TABLE = "bot_memory_index_attempt"
PERSON_REF_TABLE = "bot_person_ref"

# Identities (voices and faces) an identification store keeps.
IDENTITY_TABLE = "bot_identity"

# Prefix of provider-created vector index tables; migrations never manage these.
VECTOR_INDEX_PREFIX = "bot_memory_vec_"

# Prefix of provider-created identity print tables; disjoint from memory's, never managed by migrations.
IDENTITY_VECTOR_PREFIX = "bot_identity_vec_"

metadata = MetaData()

memory_table = Table(
    MEMORY_TABLE,
    metadata,
    Column("memory_id", Text, primary_key=True),
    Column("scope", Text, nullable=False, server_default="memory"),
    Column("context_ref", Text, nullable=True),
    Column("subject_ref", Text, nullable=True),
    Column("kind", Text, nullable=True),
    Column("sensitivity", Text, nullable=True),
    Column("retention", Text, nullable=True),
    Column("content", Text, nullable=False),
    Column("content_format", Text, nullable=False, server_default="text/plain"),
    Column("query_tags", Text, nullable=True),
    Column("metadata_json", Text, nullable=False, server_default="{}"),
    # Stored as text for dialect neutrality; providers may map to timestamptz.
    Column("created_at", Text, nullable=False, server_default=func.now()),
    Index("bot_memory_context_idx", "context_ref", "subject_ref"),
    Index("bot_memory_query_idx", "query_tags"),
    Index("bot_memory_scope_idx", "scope", "created_at"),
)

# Register behavior tables on the same MetaData so migrations (``migrations/``) track them.
from mosfet.behavior import storage  # noqa: E402, F401

_ = storage.behavior_table

# Short-term memory register: one row per admitted environment stimulus. Payload
# projections are bounded scalar fields only (identifier-like values, never media or prose);
# capacity and access-reinforced retention are enforced by the recording ability, not the schema.
stm_memory_table = Table(
    STM_MEMORY_TABLE,
    metadata,
    Column("stm_id", Text, primary_key=True),
    Column("stimulus_name", Text, nullable=False),
    Column("payload_json", Text, nullable=False, server_default="{}"),
    Column("event_id", Text, nullable=True),
    Column("source", Text, nullable=True),
    Column("target", Text, nullable=True),
    # Written by the recording ability with microsecond UTC precision so lexicographic
    # ordering matches chronological ordering across dialects that store this as text.
    Column("created_at", Text, nullable=False),
    # Reinforcement (reconsolidation): set to an access timestamp when grounding uses this
    # entry. Retention and recency read the effective clock max(created_at, last_accessed_at).
    Column("last_accessed_at", Text, nullable=True),
    Index("bot_stm_memory_recent_idx", "stimulus_name", "created_at"),
    Index("bot_stm_memory_created_idx", "created_at"),
)

# Short-term memory of turns: one row per message a Conversation committed, in words. ``seq`` is the
# write order (capacity evicts the lowest); ``at`` is microsecond UTC ISO text, so text order is
# chronological order. Text is bounded and cut at a whole word by the recording ability (``cut``);
# capacity and the recency window are enforced there too, not by the schema.
stm_turn_table = Table(
    STM_TURN_TABLE,
    metadata,
    Column("seq", Integer, primary_key=True, autoincrement=True),
    Column("turn_id", Text, nullable=False),
    Column("at", Text, nullable=False),
    Column("direction", Text, nullable=False),
    Column("speakers_json", Text, nullable=False, server_default="[]"),
    Column("text", Text, nullable=False),
    Column("cut", Integer, nullable=False, server_default="0"),
    Column("conversation", Text, nullable=False),
    Index("bot_stm_turn_at_idx", "at"),
    sqlite_autoincrement=True,
)

# One graph node. ``seq`` is the insertion order (it breaks same-instant ties and keys every
# dependent row); ``node_id`` is the opaque public id. Times are UTC ISO text at microsecond
# resolution, fixed width, so text order is chronological order.
node_table = Table(
    NODE_TABLE,
    metadata,
    Column("seq", Integer, primary_key=True, autoincrement=True),
    Column("node_id", Text, nullable=False),
    Column("kind", Text, nullable=False),
    # The person id of a person node ("self", or a minted uuid hex); the turn id an episode node began
    # with (later turns that go on with it extend it, found by this key); NULL for every other kind
    # and for an episode kept apart because its turn id was already taken. A person's refs are in
    # ``person_ref_table``.
    Column("key", Text, nullable=True),
    # A person node's text is its name, or empty while it has none.
    Column("text", Text, nullable=False, server_default=""),
    # Kind-specific typed JSON (an episode's stimulus, acts, outcome, turn id, focus, participants;
    # a person's name).
    Column("content_json", Text, nullable=False, server_default="{}"),
    Column("created_at", Text, nullable=False),
    Column("updated_at", Text, nullable=False),
    Column("recalled_at", Text, nullable=True),
    Column("recall_count", Integer, nullable=False, server_default="0"),
    Column("reviewed_at", Text, nullable=True),
    UniqueConstraint("node_id", name="bot_memory_node_id_key"),
    UniqueConstraint("kind", "key", name="bot_memory_node_kind_key"),
    Index("bot_memory_node_recent_idx", "kind", "created_at", "seq"),
    Index("bot_memory_node_review_idx", "kind", "reviewed_at", "seq"),
    sqlite_autoincrement=True,
)

# Which person each participant ref belongs to. The ref is the primary key: two persons never
# share a ref. ``origin`` is how the link was made: ``stamped`` (a ref first seen, given its own
# provisional person), ``own`` (a ref its producer stamped as the bot's own, linked to ``self``),
# or ``model`` (the bot linked it).
person_ref_table = Table(
    PERSON_REF_TABLE,
    metadata,
    Column("ref", Text, primary_key=True),
    Column("person_seq", Integer, nullable=False),
    Column("origin", Text, nullable=False),
    Column("linked_at", Text, nullable=False),
    Index("bot_person_ref_person_idx", "person_seq"),
)

part_table = Table(
    PART_TABLE,
    metadata,
    Column("node_seq", Integer, nullable=False),
    Column("ordinal", Integer, nullable=False),
    Column("kind", Text, nullable=False),
    Column("text", Text, nullable=True),
    Column("media_type", Text, nullable=True),
    Column("width_px", Integer, nullable=True),
    Column("height_px", Integer, nullable=True),
    Column("content", LargeBinary, nullable=True),
    PrimaryKeyConstraint("node_seq", "ordinal", name="bot_memory_part_pkey"),
)

edge_table = Table(
    EDGE_TABLE,
    metadata,
    Column("source_seq", Integer, nullable=False),
    Column("target_seq", Integer, nullable=False),
    Column("relation", Text, nullable=False),
    Column("weight", Float, nullable=False, server_default="1.0"),
    Column("created_at", Text, nullable=False),
    # ``stamped`` when the producer's data implied it (an episode involves a person), ``model``
    # when consolidation chose it.
    Column("origin", Text, nullable=False),
    PrimaryKeyConstraint("source_seq", "target_seq", "relation", name="bot_memory_edge_pkey"),
    Index("bot_memory_edge_target_idx", "target_seq"),
)

# The canonical vector of a node in one embedding space (little-endian float32). A vector
# index is only an accelerator and can be rebuilt from these rows.
vector_table = Table(
    VECTOR_TABLE,
    metadata,
    Column("node_seq", Integer, nullable=False),
    Column("space_model", Text, nullable=False),
    Column("dimensions", Integer, nullable=False),
    Column("embedding", LargeBinary, nullable=False),
    PrimaryKeyConstraint("node_seq", "space_model", "dimensions", name="bot_memory_vector_pkey"),
)

# The last failed attempt to index a node in one space. ``permanent`` attempts (the item can
# never be embedded in that space) are never selected again; transient ones are retried, at
# most once per backfill pass (``pass_id``).
index_attempt_table = Table(
    INDEX_ATTEMPT_TABLE,
    metadata,
    Column("node_seq", Integer, nullable=False),
    Column("space_model", Text, nullable=False),
    Column("dimensions", Integer, nullable=False),
    Column("failure_kind", Text, nullable=False),
    Column("permanent", Integer, nullable=False),
    Column("pass_id", Text, nullable=True),
    Column("attempted_at", Text, nullable=False),
    PrimaryKeyConstraint("node_seq", "space_model", "dimensions", name="bot_memory_index_attempt_pkey"),
    Index("bot_memory_index_attempt_space_idx", "space_model", "dimensions", "permanent"),
)

# One identity of one kind. ``seq`` is SQLite's never-reused row number and the rowid of the
# identity's print in its model's provider vector table; ``id`` is the public id (minted or
# declared). ``seen_count`` is the running-mean weight of the print, capped by the match policy.
# ``linked`` is 1 while a registry's link mark says the id belongs to a person (memory holds which
# person); own and linked ids are never pruned. Times
# are fixed-width UTC ISO text at microsecond resolution, so text order is chronological order.
identity_table = Table(
    IDENTITY_TABLE,
    metadata,
    Column("seq", Integer, primary_key=True, autoincrement=True),
    Column("kind", Text, nullable=False),
    Column("id", Text, nullable=False),
    Column("model", Text, nullable=False),
    Column("own", Integer, nullable=False),
    Column("linked", Integer, nullable=False, server_default="0"),
    Column("created_at", Text, nullable=False),
    Column("seen_at", Text, nullable=False),
    Column("seen_count", Integer, nullable=False),
    UniqueConstraint("kind", "id", name="bot_identity_kind_id_key"),
    Index("bot_identity_prune_idx", "kind", "own", "linked", "seen_at"),
    sqlite_autoincrement=True,
)

__all__ = [
    "EDGE_TABLE",
    "IDENTITY_TABLE",
    "IDENTITY_VECTOR_PREFIX",
    "INDEX_ATTEMPT_TABLE",
    "MEMORY_TABLE",
    "NODE_TABLE",
    "PART_TABLE",
    "PERSON_REF_TABLE",
    "STM_MEMORY_TABLE",
    "STM_TURN_TABLE",
    "VECTOR_INDEX_PREFIX",
    "VECTOR_TABLE",
    "edge_table",
    "identity_table",
    "index_attempt_table",
    "memory_table",
    "metadata",
    "node_table",
    "part_table",
    "person_ref_table",
    "stm_memory_table",
    "stm_turn_table",
    "vector_table",
]
