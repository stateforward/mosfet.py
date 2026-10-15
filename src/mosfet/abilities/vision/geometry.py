"""Where something is in an image, independent of the image's size or the model that found it."""

import typing

import pydantic

_SLACK: typing.Final = 1e-6
"""Rounding a producer's pixel box to fractions may overshoot an edge by this much."""


class Box(pydantic.BaseModel):
    """An axis-aligned box in normalized image coordinates: 0..1 on both axes, origin at the top left.

    Producers convert their own pixel or model conventions to a ``Box``; consumers never see pixels.
    """

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": (
                "Where something is in the image, as fractions of the image's width and height: x and y are "
                "the top-left corner (0,0 is the image's top-left), width and height the box's size."
            ),
            "examples": [{"x": 0.25, "y": 0.1, "width": 0.2, "height": 0.35}],
        },
    )

    x: float = pydantic.Field(
        ge=0.0, le=1.0, description="Left edge, as a fraction of the image width.", examples=[0.25]
    )
    y: float = pydantic.Field(
        ge=0.0, le=1.0, description="Top edge, as a fraction of the image height.", examples=[0.1]
    )
    width: float = pydantic.Field(
        gt=0.0, le=1.0, description="Box width, as a fraction of the image width.", examples=[0.2]
    )
    height: float = pydantic.Field(
        gt=0.0, le=1.0, description="Box height, as a fraction of the image height.", examples=[0.35]
    )

    @pydantic.model_validator(mode="after")
    def _inside_the_image(self) -> typing.Self:
        if self.x + self.width > 1.0 + _SLACK:
            raise ValueError("a box cannot extend past the image's right edge")
        if self.y + self.height > 1.0 + _SLACK:
            raise ValueError("a box cannot extend past the image's bottom edge")
        return self


__all__ = ["Box"]
