"""Single source of truth for "is this HSM instance started?".

``stateforward-hsm`` (pinned ``>=1.3.2,<1.4``) exposes no predicate for this. ``hsm.started``
is a constructor (``New`` + ``Start``), not a question, and the only observable that separates
a started machine from an unstarted or stopped one is ``Instance.take_snapshot()``:

* never handed to ``hsm.new`` — ``Instance.take_snapshot`` returns an empty ``hsm.Snapshot``;
* started — returns a populated snapshot;
* stopped — ``HSM.take_snapshot`` raises ``"take snapshot requires a started HSM"``.

BOTH arms of the ``except`` below are load-bearing — do not delete either. The exception
*type* for that third case changes across the supported range, verified against the sdists
on PyPI:

* 1.3.2 raises ``hsm.ErrorValidatingModel`` (``hsm.py`` lines 4166 and 4757); it has no
  ``_runtime_error`` helper at all.
* 1.3.3 raises a plain ``RuntimeError`` built by ``_runtime_error`` (``hsm.py`` line 131)
  at ``hsm.py`` lines 4161 and 4748.

``ErrorValidatingModel`` derives from ``Exception``, not ``RuntimeError``, so neither arm
subsumes the other. ``uv.lock`` currently resolves 1.3.2, which means on a locked checkout
the ``RuntimeError`` arm never fires and looks like removable dead code — it is not, because
``pyproject.toml`` permits ``>=1.3.2,<1.4`` and 1.3.3 is published. Re-verify with a scratch
install of each version before touching this.

Both versions carry the same message, so the message — not the type — is the only reliable
discriminator. A bare ``except RuntimeError`` here would swallow unrelated runtime failures
raised from ``take_snapshot`` overrides or HSM callbacks and report them as "not started";
hence the message check plus re-raise.

This is deliberately the only place in the tree that inspects ``hsm`` exception prose for
liveness. It answers a question about a machine's own lifecycle; it is not a substitute for
``hsm.Context.is_done()``, ``instance.state()`` probing, or delivery gating — dispatch is
still the gate (HSM-DELIVERY-001), and cross-machine coordination stays on typed events.
"""

import hsm

_NOT_STARTED = "take snapshot requires a started HSM"


def snapshot_if_started(instance: hsm.Instance) -> hsm.Snapshot | None:
    """``instance``'s snapshot when it has a running HSM, else ``None`` — one ``take_snapshot`` call.

    The liveness check and the snapshot a caller wants afterward both come from the same
    ``take_snapshot()`` call, which is not free: a caller that needs both (e.g. folding a
    peripheral's own attributes only when it is live) should call this once rather than
    probing with :func:`is_started` and then snapshotting again.
    """

    try:
        snapshot = instance.take_snapshot()
    except (hsm.ErrorValidatingModel, RuntimeError) as error:
        if _NOT_STARTED not in str(error):
            raise
        return None
    # Mirrors hsm.TakeSnapshot: an all-empty snapshot means the instance was never started.
    if not (snapshot.ID or snapshot.QualifiedName or snapshot.State):
        return None
    return snapshot


def is_started(instance: hsm.Instance) -> bool:
    """True when ``instance`` has a running HSM and can therefore be addressed by ``hsm.id``."""

    return snapshot_if_started(instance) is not None


# Lifecycle idempotency for attach/detach/activate (not peer-state gating; HSM-CONTEXT-001).
# Prefer typed HSM errors when present. Stock stateforward-hsm still often surfaces these
# conditions as fixed exception prose (ErrorAlreadyStarted / ErrorMissingHSM are exported
# but not always raised). All residual message detection is confined to these two predicates
# — call sites must not open-code hsm exception text. Centralized here (single source of
# truth) so body, devices, and abilities share one prose allowlist that may only shrink.


def is_already_running_error(error: BaseException) -> bool:
    """True when ``error`` means the machine already has a running HSM (idempotent start)."""

    if isinstance(error, hsm.ErrorAlreadyStarted):
        return True
    if isinstance(error, hsm.ErrorValidatingModel | RuntimeError):
        message = str(error)
        return "already has a running HSM" in message or "already started HSM" in message
    return False


def is_not_started_error(error: BaseException) -> bool:
    """True when ``error`` means the machine has no running HSM (idempotent stop/dispatch)."""

    if isinstance(error, hsm.ErrorMissingHSM):
        return True
    if isinstance(error, RuntimeError):
        message = str(error)
        return (
            "dispatch requires a started HSM" in message
            or "take snapshot requires a started HSM" in message
            or "operation requires a started HSM" in message
            or "restart requires a started HSM" in message
            or "set requires a started HSM" in message
        )
    return False
