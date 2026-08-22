from bot import abilities
from bot.protocols import attachment

import asyncio
import datetime
import importlib.util
import typing
import weakref

import hsm
import bot
import pytest


class EchoEncoder(abilities.Encoder[str, bytes]):
    @typing.override
    async def encode(self, input: str) -> bytes:
        return input.encode()


class EchoDecoder(abilities.Decoder[bytes, str]):
    @typing.override
    async def decode(self, input: bytes) -> str:
        return input.decode()


class EchoGenerator(abilities.Generator[str, str]):
    @typing.override
    async def generate(self, input: str) -> str:
        return f"generated:{input}"


class EchoClassifier(abilities.Classifier[str, bool]):
    @typing.override
    async def classify(self, input: str) -> bool:
        return input == "voice"


class FailingEncoder(abilities.Encoder[str, bytes]):
    @typing.override
    async def encode(self, input: str) -> bytes:
        del input
        raise RuntimeError("encoding failed")


class AbilityOwner(hsm.Instance):
    model: typing.ClassVar[hsm.Model] = bot.define(
        "GenericAbilityOwner",
        hsm.initial(hsm.target("active")),
        hsm.state("active"),
    )


async def start_directed_child(child: abilities.Ability[typing.Any, typing.Any]) -> hsm.Context:
    ctx = hsm.Context().with_value(
        hsm.Keys.Instances,
        weakref.WeakValueDictionary[object, hsm.Instance](),
    )
    owner = AbilityOwner()
    _ = await bot.started(ctx, owner, owner.model)
    await child.attach(
        ctx,
        attachment.AttachEvent.with_data(attachment.AttachData(actor=owner)),
    )
    return ctx


class RecordingEncoding(abilities.Encoding[str, bytes]):
    def __init__(self, *, encoder: abilities.Encoder[str, bytes]) -> None:
        super().__init__(encoder=encoder)


class RecordingDecoding(abilities.Decoding[bytes, str]):
    def __init__(self, *, decoder: abilities.Decoder[bytes, str]) -> None:
        super().__init__(decoder=decoder)


class RecordingGenerative(abilities.Generative[str, str]):
    def __init__(self, *, generator: abilities.Generator[str, str]) -> None:
        super().__init__(generator=generator)


class RecordingClassifying(abilities.Classifying[str, bool]):
    def __init__(self, *, classifier: abilities.Classifier[str, bool]) -> None:
        super().__init__(classifier=classifier)


def test_abilities_export_encoding_decoding_generative_and_classifying_contracts() -> None:
    encoder = EchoEncoder()
    decoder = EchoDecoder()
    generator = EchoGenerator()
    classifier = EchoClassifier()

    encoding = RecordingEncoding(encoder=encoder)
    decoding = RecordingDecoding(decoder=decoder)
    generative = RecordingGenerative(generator=generator)
    classifying = RecordingClassifying(classifier=classifier)

    assert encoding.encoder is encoder
    assert decoding.decoder is decoder
    assert generative.generator is generator
    assert classifying.classifier is classifier
    assert abilities.Encoding.model is not None
    assert abilities.Decoding.model is not None
    assert abilities.Generative.model is not None
    assert abilities.Classifying.model is not None


def test_legacy_base_ability_package_is_removed() -> None:
    legacy_packages = (
        "bot.abilities.generating",
        "bot.abilities.base",
    )

    for package in legacy_packages:
        assert importlib.util.find_spec(package) is None


@pytest.mark.parametrize(
    ("child", "request_data", "expected"),
    [
        (RecordingEncoding(encoder=EchoEncoder()), "hello", b"hello"),
        (RecordingDecoding(decoder=EchoDecoder()), b"hello", "hello"),
        (RecordingGenerative(generator=EchoGenerator()), "hello", "generated:hello"),
        (RecordingClassifying(classifier=EchoClassifier()), "voice", True),
    ],
)
def test_generic_ability_directed_success_returns_to_terminal_operation(
    child: abilities.Ability[typing.Any, typing.Any],
    request_data: object,
    expected: object,
) -> None:
    async def run() -> hsm.Event[typing.Any]:
        ctx = await start_directed_child(child)
        return await abilities.run_terminal_operation(
            ctx,
            child=child,
            request=child.input_event.with_data_and_id(request_data, "directed-success"),
            terminals=(child.output_event, child.failed_event),
            timeout=datetime.timedelta(milliseconds=100),
        )

    terminal = asyncio.run(run())

    assert terminal.id == "directed-success"
    assert terminal.data == expected


def test_generic_ability_directed_failure_returns_to_terminal_operation() -> None:
    async def run() -> hsm.Event[typing.Any]:
        child = RecordingEncoding(encoder=FailingEncoder())
        ctx = await start_directed_child(child)
        return await abilities.run_terminal_operation(
            ctx,
            child=child,
            request=child.input_event.with_data_and_id("hello", "directed-failure"),
            terminals=(child.output_event, child.failed_event),
            timeout=datetime.timedelta(milliseconds=100),
        )

    terminal = asyncio.run(run())

    assert terminal.id == "directed-failure"
    assert isinstance(terminal.data, abilities.FailureData)
    assert terminal.data.message == "encoding failed"


def test_generic_ability_sequential_directed_operations_keep_distinct_ids() -> None:
    async def run() -> tuple[hsm.Event[typing.Any], hsm.Event[typing.Any]]:
        child = RecordingGenerative(generator=EchoGenerator())
        ctx = await start_directed_child(child)

        async def operation(operation_id: str) -> hsm.Event[typing.Any]:
            return await abilities.run_terminal_operation(
                ctx,
                child=child,
                request=child.input_event.with_data_and_id(operation_id, operation_id),
                terminals=(child.output_event, child.failed_event),
                timeout=datetime.timedelta(milliseconds=100),
            )

        return await operation("first"), await operation("second")

    first, second = asyncio.run(run())

    assert (first.id, first.data) == ("first", "generated:first")
    assert (second.id, second.data) == ("second", "generated:second")
