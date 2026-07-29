from .events import OutputEvent, AudioOutputData, SoundProvenanceData, SoundProvenanceEvent, routed_audio_event

import collections.abc
import dataclasses
import typing

import hsm

from bot.device import Device
from bot.environment import SoundData, SoundEvent, Environment, space


class Speaker(Device):
    """Transducer that converts audio output signal into acoustic energy in the environment.

    The mirror of :class:`~bot.devices.audio.microphone.Microphone`: a microphone converts what
    it hears into signal on the way in, a speaker converts signal into sound on the way out.
    Both are transducers with no firmware and no routing decision of their own; the difference is
    only which way the attachment points. The controller that attached decides what to play.

    Transduction happens only while attached, exactly as an unwired speaker is silent.
    """

    output_event: typing.ClassVar[hsm.Event[AudioOutputData]] = OutputEvent

    _amplitude_db: float | None
    _sound_owner: str | None

    def __init__(
        self,
        *,
        peripherals: collections.abc.Iterable[Device] = (),
        placement: space.Placement | None = None,
        amplitude_db: float | None = None,
    ) -> None:
        super().__init__(peripherals=peripherals, placement=placement)
        # How loud this transducer plays, measured at the reference distance. A handset earpiece
        # and a room speaker are different hardware; without a value the sound carries everywhere,
        # which is what keeps geometry opt-in.
        self._amplitude_db = amplitude_db
        # Who holds this transducer, told to it by the device that wired it in
        # (devices.audio.sound_provenance). Unlabeled means the sounds say so: null, never invented.
        self._sound_owner = None

    @staticmethod
    def _note_sound_provenance(ctx: hsm.Context, instance: "Speaker", event: hsm.Event[typing.Any]) -> None:
        """Label this transducer with the holder its owning device just declared."""

        del ctx
        data = event.data
        assert isinstance(data, SoundProvenanceData)
        instance._sound_owner = data.owner

    @staticmethod
    def _transduce(ctx: hsm.Context, instance: "Speaker", event: hsm.Event[typing.Any]) -> None:
        """Convert signal from an attached controller into acoustic energy in the environment.

        Environment broadcast is the input elevation path: bots fan ``environment.sound`` to input abilities
        and never treat raw device playout as cognition input. Targeted playout into another
        device still uses ``devices.audio.output`` via :meth:`dispatch_audio_output`.
        """

        data = event.data
        if not isinstance(data, AudioOutputData):
            return
        placement = instance._placement
        sound = dataclasses.replace(
            SoundEvent.with_data(
                SoundData(
                    audio=data.audio,
                    media_type=data.media_type,
                    sample_rate_hz=data.sample_rate_hz,
                    channels=data.channels,
                    amplitude_db=instance._amplitude_db,
                    owner=instance._sound_owner,
                )
            ),
            source=hsm.id(instance),
            metadata=dict(event.metadata),
        )
        # The environment works out what reaches whom; this only says how loud, and from where.
        _ = Environment.from_context(ctx).broadcast(sound, origin=None if placement is None else placement.position)

    model: typing.ClassVar[hsm.Model | None] = hsm.redefine(
        typing.cast(hsm.Model, Device.model),
        hsm.transition(hsm.source("attached"), hsm.on(OutputEvent), hsm.effect(_transduce)),
        hsm.transition(hsm.source("attached"), hsm.on(SoundProvenanceEvent), hsm.effect(_note_sound_provenance)),
    )

    def dispatch_audio_output(
        self,
        ctx: hsm.Context,
        target: hsm.Instance,
        data: AudioOutputData,
        *,
        metadata: collections.abc.Mapping[str, object] | None = None,
    ) -> collections.abc.Awaitable[None]:
        event = routed_audio_event(self.output_event, data, source=self, target=target, metadata=metadata)
        return target.dispatch(ctx, event)
