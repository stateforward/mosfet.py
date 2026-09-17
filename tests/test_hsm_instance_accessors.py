from __future__ import annotations

import ast
import pathlib


_ROOT = pathlib.Path(__file__).resolve().parents[1]
_ALLOWED_NON_HSM_PROPERTIES = {
    "src/providers/livekit/src/mosfet/providers/livekit/audio.py:AudioFrame.data",
    "src/providers/livekit/src/mosfet/providers/livekit/audio.py:AudioFrame.num_channels",
    "src/providers/livekit/src/mosfet/providers/livekit/audio.py:AudioFrame.sample_rate",
    "src/providers/livekit/src/mosfet/providers/livekit/audio.py:AudioFrame.samples_per_channel",
    "src/providers/livekit/src/mosfet/providers/livekit/room_audio.py:RoomHandle.local_participant",
    "src/providers/livekit/src/mosfet/providers/livekit/room_audio.py:TrackPublication.sid",
    "src/mosfet/abilities/communication/conversation/conversation.py:Conversation.detector_count",
    "src/mosfet/abilities/communication/conversation/conversation.py:Conversation.detector_refs",
    "src/mosfet/abilities/communication/conversation/conversation.py:Conversation.encoding",
    "src/mosfet/abilities/communication/conversation/conversation.py:Conversation.memory",
    "src/mosfet/abilities/communication/conversation/conversation.py:Conversation.session_refs",
    "src/mosfet/abilities/communication/conversation/conversation.py:Conversation.typing",
    "src/mosfet/abilities/communication/conversation/turn_detector/turn_detector.py:TurnDetector.decoder",
    "src/mosfet/abilities/hearing/sound/classification.py:OutputData.is_labeled",
    "src/mosfet/abilities/hearing/voice/segment.py:VoiceSegment.duration_seconds",
    "src/mosfet/abilities/listening/interpretation.py:SpeechData.duration_seconds",
    "src/mosfet/behavior/diagnostic.py:Checked.ok",
    "src/mosfet/behavior/diagnostic.py:Report.errors",
    "src/mosfet/behavior/diagnostic.py:Report.ok",
    "src/mosfet/devices/audio/speaker.py:Speaker.amplitude_db",
    "src/mosfet/devices/audio/speaker.py:Speaker.placement",
    "src/mosfet/protocols/yamux/frame.py:FrameData.is_ack",
    "src/mosfet/protocols/yamux/frame.py:FrameData.is_fin",
    "src/mosfet/protocols/yamux/frame.py:FrameData.is_rst",
    "src/mosfet/protocols/yamux/frame.py:FrameData.is_syn",
    "src/mosfet/environment/environment.py:Environment.environment_id",
    "src/mosfet/environment/environment.py:Environment.scope_path",
    "src/mosfet/devices/phone/phone.py:EventRecorder.events",
}
# HSM instances may own private runtime data, including mutable mappings.
# This guard blocks public member surfaces that invite direct caller mutation
# instead of modeled entry, exit, effect, or activity behavior.
_FORBIDDEN_PUBLIC_HSM_OWNED_MEMBER_ANNOTATIONS = {
    "src/mosfet/abilities/ability.py:Ability.abilities",
    "src/mosfet/abilities/listening/interpretation.py:Interpretation.sound_classification",
    "src/mosfet/abilities/listening/interpretation.py:Interpretation.speech_decoder",
    "src/mosfet/abilities/listening/interpretation.py:Interpretation.speech_decoding",
    "src/mosfet/abilities/listening/interpretation.py:Interpretation.voice_detection",
    "src/mosfet/abilities/listening/interpretation.py:Interpretation.voice_activity_classifier",
    "src/mosfet/abilities/listening/interpretation.py:Interpretation.voice_diarization",
    "src/mosfet/abilities/listening/interpretation.py:Interpretation.voice_diarizer",
    "src/mosfet/abilities/listening/listening.py:Listening.interpretation",
    "src/mosfet/abilities/listening/listening.py:Listening.operation_timeout",
    "src/mosfet/abilities/listening/listening.py:Listening.sensitivity",
    "src/mosfet/abilities/communication/conversation/turn_detector/turn_detector.py:TurnDetector.listening",
    "src/mosfet/abilities/communication/conversation/turn_detector/turn_detector.py:TurnDetector.reading",
    "src/mosfet/abilities/reading/reading.py:Reading.image_decoder",
    "src/mosfet/abilities/reading/reading.py:Reading.image_decoding",
    "src/mosfet/abilities/reading/reading.py:Reading.operation_timeout",
    "src/mosfet/abilities/reading/reading.py:Reading.output_encoder",
    "src/mosfet/abilities/reading/reading.py:Reading.output_encoding",
    "src/mosfet/abilities/reading/reading.py:Reading.text_decoder",
    "src/mosfet/abilities/reading/reading.py:Reading.text_decoding",
    "src/mosfet/abilities/reading/reading.py:Reading.visual_classification",
    "src/mosfet/abilities/reading/reading.py:Reading.visual_classifier",
    "src/mosfet/behavior/behavior.py:Behavior.callback_runtime",
    "src/mosfet/behavior/behavior.py:Behavior.spec",
    "src/providers/livekit/src/mosfet/providers/livekit/phone.py:Service.active_call_id",
    "src/providers/livekit/src/mosfet/providers/livekit/phone.py:Service.active_transfer_id",
    "src/providers/livekit/src/mosfet/providers/livekit/phone.py:Service.active_transfer_target",
    "src/providers/livekit/src/mosfet/providers/livekit/phone.py:Service.phone_event_target",
    "src/providers/livekit/src/mosfet/providers/livekit/room_audio.py:RoomAudioTrackPath.local_track_sid",
    "src/providers/livekit/src/mosfet/providers/livekit/room_audio.py:RoomAudioTrackPath.remote_audio_bytes",
    "src/providers/livekit/src/mosfet/providers/livekit/room_audio.py:RoomAudioTrackPath.remote_audio_chunks",
    "src/mosfet/bot.py:Bot.focused_device",
    "src/mosfet/devices/phone/phone.py:Phone.firmware_instance",
    "src/mosfet/devices/phone/phone.py:Firmware.closed_call_ids",
    "src/mosfet/devices/phone/phone.py:Firmware.current_call_id",
    "src/mosfet/devices/phone/phone.py:Firmware.current_transfer_id",
    "src/mosfet/devices/phone/phone.py:Firmware.current_transfer_target",
}
_FORBIDDEN_PUBLIC_HSM_OWNED_HELPER_FUNCTIONS = {
    "src/providers/livekit/src/mosfet/providers/livekit/phone.py:active_call_id",
    "src/providers/livekit/src/mosfet/providers/livekit/phone.py:active_transfer_id",
    "src/providers/livekit/src/mosfet/providers/livekit/phone.py:active_transfer_target",
    "src/mosfet/abilities/ability.py:ability_children",
    "src/mosfet/devices/phone/phone.py:current_call_id",
    "src/mosfet/devices/phone/phone.py:current_transfer_id",
    "src/mosfet/devices/phone/phone.py:current_transfer_target",
}
_FORBIDDEN_HSM_OWNED_STATE_EXPORTS = {
    "src/mosfet/abilities/ability.py:_ability_children",
}
_HSM_INSTANCE_STATE_VAR_NAMES = {"instance", "ability", "bot", "phone", "firmware", "device"}


def _has_property_decorator(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    return any(isinstance(decorator, ast.Name) and decorator.id == "property" for decorator in node.decorator_list)


def _production_property_accessors() -> set[str]:
    accessors: set[str] = set()
    for root_name in ("src", "examples/phone_bot/src"):
        for path in (_ROOT / root_name).rglob("*.py"):
            tree = ast.parse(path.read_text(), filename=str(path))
            relative = path.relative_to(_ROOT).as_posix()
            for class_node in (node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)):
                for member in class_node.body:
                    if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)) and _has_property_decorator(member):
                        accessors.add(f"{relative}:{class_node.name}.{member.name}")
    return accessors


def _production_class_member_annotations() -> set[str]:
    annotations: set[str] = set()
    for root_name in ("src", "examples/phone_bot/src"):
        for path in (_ROOT / root_name).rglob("*.py"):
            tree = ast.parse(path.read_text(), filename=str(path))
            relative = path.relative_to(_ROOT).as_posix()
            for class_node in (node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)):
                for member in class_node.body:
                    if isinstance(member, ast.AnnAssign) and isinstance(member.target, ast.Name):
                        annotations.add(f"{relative}:{class_node.name}.{member.target.id}")
    return annotations


def _production_function_definitions() -> set[str]:
    definitions: set[str] = set()
    for root_name in ("src", "examples/phone_bot/src"):
        for path in (_ROOT / root_name).rglob("*.py"):
            tree = ast.parse(path.read_text(), filename=str(path))
            relative = path.relative_to(_ROOT).as_posix()
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    definitions.add(f"{relative}:{node.name}")
    return definitions


def _production_hsm_instance_vars_reads() -> set[str]:
    reads: set[str] = set()
    for root_name in ("src", "examples/phone_bot/src"):
        for path in (_ROOT / root_name).rglob("*.py"):
            tree = ast.parse(path.read_text(), filename=str(path))
            relative = path.relative_to(_ROOT).as_posix()
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "vars"
                    and len(node.args) == 1
                    and isinstance(node.args[0], ast.Name)
                    and node.args[0].id in _HSM_INSTANCE_STATE_VAR_NAMES
                ):
                    reads.add(f"{relative}:{node.lineno}:{node.args[0].id}")
    return reads


def _production_all_exports() -> set[str]:
    exports: set[str] = set()
    for root_name in ("src", "examples/phone_bot/src"):
        for path in (_ROOT / root_name).rglob("*.py"):
            tree = ast.parse(path.read_text(), filename=str(path))
            relative = path.relative_to(_ROOT).as_posix()
            for node in ast.walk(tree):
                if not isinstance(node, ast.Assign):
                    continue
                if not any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets):
                    continue
                if not isinstance(node.value, (ast.List, ast.Tuple)):
                    continue
                for element in node.value.elts:
                    if isinstance(element, ast.Constant) and isinstance(element.value, str):
                        exports.add(f"{relative}:{element.value}")
    return exports


def test_hsm_instances_do_not_expose_property_accessors() -> None:
    assert _production_property_accessors() == _ALLOWED_NON_HSM_PROPERTIES


def test_hsm_owned_members_are_not_exposed_as_public_class_annotations() -> None:
    assert _production_class_member_annotations().isdisjoint(_FORBIDDEN_PUBLIC_HSM_OWNED_MEMBER_ANNOTATIONS)


def test_hsm_owned_members_are_not_exposed_through_public_helper_functions() -> None:
    assert _production_function_definitions().isdisjoint(_FORBIDDEN_PUBLIC_HSM_OWNED_HELPER_FUNCTIONS)


def test_hsm_owned_state_helpers_are_not_exported() -> None:
    assert _production_all_exports().isdisjoint(_FORBIDDEN_HSM_OWNED_STATE_EXPORTS)


def test_hsm_callbacks_do_not_read_instance_state_through_vars() -> None:
    assert _production_hsm_instance_vars_reads() == set()
