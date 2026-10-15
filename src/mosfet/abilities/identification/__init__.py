"""One provider-agnostic identity contract for voices and faces.

Usage::

    from mosfet.abilities import identification

    policy = identification.MatchPolicy(threshold=0.50, margin=0.05, refine_threshold=0.65)
    store = identification.InProcessStore(max_ids=1000)
    registry = identification.Registry(
        identifier=identification.Matcher(store=store, policy=policy),
        kind="voice",
        max_unlinked_age=datetime.timedelta(days=30),
        max_ids=1000,
    )

The perceiving ability owns the ``Registry`` and sends it ``IdentifyEvent`` (answered
``IdentifiedEvent``), ``EnrollEvent`` (answered ``EnrolledEvent``) and ``ForgetEvent`` (answered
``ForgottenEvent``); any request may instead be answered ``FailedEvent`` with a closed
``FailureKind``. Producers stamp what they recognized with ``participant(kind, id)``. A store
implements ``Store``; a remote service implements ``Identifier``.
"""

from .identification import (
    Decision,
    EnrollData,
    EnrollEvent,
    EnrolledEvent,
    FailedEvent,
    FailureData,
    FailureKind,
    ForgetData,
    ForgetEvent,
    ForgottenData,
    ForgottenEvent,
    Id,
    IdentifiedData,
    IdentifiedEvent,
    Identifier,
    IdentifyData,
    IdentifyEvent,
    Kind,
    Match,
    MatchPolicy,
    Mint,
    Neighbor,
    Print,
    Recognition,
    Ref,
    Refine,
    Registry,
    StoreError,
    decide,
    participant,
    ref,
)
from .store import InProcessStore, Matcher, Store

__all__ = [
    "Decision",
    "EnrollData",
    "EnrollEvent",
    "EnrolledEvent",
    "FailedEvent",
    "FailureData",
    "FailureKind",
    "ForgetData",
    "ForgetEvent",
    "ForgottenData",
    "ForgottenEvent",
    "Id",
    "IdentifiedData",
    "IdentifiedEvent",
    "Identifier",
    "IdentifyData",
    "IdentifyEvent",
    "InProcessStore",
    "Kind",
    "Match",
    "MatchPolicy",
    "Matcher",
    "Mint",
    "Neighbor",
    "Print",
    "Recognition",
    "Ref",
    "Refine",
    "Registry",
    "Store",
    "StoreError",
    "decide",
    "participant",
    "ref",
]
