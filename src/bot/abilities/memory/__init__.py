"""Memory primitives for stateforward.bot agents.

Canonical schema is SQLAlchemy Core (``memory_table`` / ``metadata``). Domain code builds
Core clauses, compiles them with ``compile_statement(s)`` into crude ``Statement(sql, parameters)``,
and applies a transaction through the ``Memory`` ability events. Generation is a sibling ability
that produces content for INSERT—not a store encode/decode pipeline.
"""

from . import (
    associative,
    classification,
    consolidation,
    long_term,
    memory,
    payload,
    schema,
    short_term,
    stm,
    store,
)
from bot.abilities.memory.associative import AssociativeMemory
from bot.abilities.memory.classification import (
    EncodedMemory,
    GeneratedMemory,
    MemoryClassification,
    MemoryClassificationKind,
    MemoryClassificationRetention,
    MemoryClassificationSensitivity,
    MemoryClassifier,
)
from bot.abilities.memory.consolidation import (
    MemoryConsolidation,
    MemoryConsolidationGenerator,
    MemoryConsolidationSource,
)
from bot.abilities.memory.long_term import LongTermMemory
from bot.abilities.memory.memory import (
    CandidateData,
    InputData,
    Memory,
    MemoryGeneration,
    MemoryGenerator,
    OutputData,
    SourceData,
    Statement,
    StatementResult,
    memory_model,
)
from bot.abilities.memory.schema import STM_EVENTS_TABLE, memory_table, metadata, stm_events_table
from bot.abilities.memory.stm import ObservedEvent, RecordData, StmEventMemory
from bot.abilities.memory.store import compile_statement, compile_statements
from bot.abilities.memory.payload import (
    DecodeData,
    DecodedData,
    EncodeData,
    MemoryMetadataValue,
    MemoryModality,
    MemoryPart,
    MemoryVector,
)
from bot.abilities.memory.short_term import ShortTermMemory
from bot.abilities.memory.store import (
    MEMORY_TABLE,
    MemoryRecord,
    MemoryStore,
    ParameterValue,
    Row,
)

__all__ = [
    "MEMORY_TABLE",
    "STM_EVENTS_TABLE",
    "AssociativeMemory",
    "CandidateData",
    "DecodeData",
    "DecodedData",
    "EncodeData",
    "EncodedMemory",
    "GeneratedMemory",
    "InputData",
    "LongTermMemory",
    "Memory",
    "MemoryClassification",
    "MemoryClassificationKind",
    "MemoryClassificationRetention",
    "MemoryClassificationSensitivity",
    "MemoryClassifier",
    "MemoryConsolidation",
    "MemoryConsolidationGenerator",
    "MemoryConsolidationSource",
    "MemoryGeneration",
    "MemoryGenerator",
    "MemoryMetadataValue",
    "MemoryModality",
    "MemoryPart",
    "MemoryRecord",
    "MemoryStore",
    "MemoryVector",
    "ObservedEvent",
    "OutputData",
    "ParameterValue",
    "RecordData",
    "Row",
    "ShortTermMemory",
    "StmEventMemory",
    "SourceData",
    "Statement",
    "StatementResult",
    "associative",
    "classification",
    "compile_statement",
    "compile_statements",
    "consolidation",
    "long_term",
    "memory",
    "memory_model",
    "memory_table",
    "metadata",
    "payload",
    "schema",
    "short_term",
    "stm",
    "stm_events_table",
    "store",
]
