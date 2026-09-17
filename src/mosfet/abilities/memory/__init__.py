"""Memory primitives for stateforward.mosfet agents.

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
from mosfet.abilities.memory.associative import AssociativeMemory
from mosfet.abilities.memory.classification import (
    EncodedMemory,
    GeneratedMemory,
    MemoryClassification,
    MemoryClassificationKind,
    MemoryClassificationRetention,
    MemoryClassificationSensitivity,
    MemoryClassifier,
)
from mosfet.abilities.memory.consolidation import (
    MemoryConsolidation,
    MemoryConsolidationGenerator,
    MemoryConsolidationSource,
)
from mosfet.abilities.memory.long_term import LongTermMemory
from mosfet.abilities.memory.memory import (
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
from mosfet.abilities.memory.schema import STM_MEMORY_TABLE, memory_table, metadata, stm_memory_table
from mosfet.abilities.memory.stm import ObservedEvent, RecordData, StmMemory
from mosfet.abilities.memory.store import compile_statement, compile_statements
from mosfet.abilities.memory.payload import (
    DecodeData,
    DecodedData,
    EncodeData,
    MemoryMetadataValue,
    MemoryModality,
    MemoryPart,
    MemoryVector,
)
from mosfet.abilities.memory.short_term import ShortTermMemory
from mosfet.abilities.memory.store import (
    MEMORY_TABLE,
    MemoryRecord,
    MemoryStore,
    ParameterValue,
    Row,
)

__all__ = [
    "MEMORY_TABLE",
    "STM_MEMORY_TABLE",
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
    "StmMemory",
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
    "stm_memory_table",
    "store",
]
