#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "stateforward.bot",
# ]
# [tool.uv.sources]
# stateforward-bot = { path = "..", editable = true }
# ///
"""Print a small stateforward.bot notification schema example."""

import json

from bot.device import DeviceNotificationData, Notification


def main() -> None:
    data = DeviceNotificationData(
        device_qualified_name="device/operator-phone",
        firmware_qualified_name="phone.caller",
        service_qualified_name="phone.webrtc",
        hint="answer_call",
    )
    notification = Notification(data=data)
    schema = DeviceNotificationData.model_json_schema()

    summary: dict[str, object] = {
        "notification_model": notification.model.qualified_name,
        "notification_data": data.model_dump(),
        "required_fields": schema.get("required", []),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
