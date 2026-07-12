from bot import abilities

import importlib.util
import typing

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
