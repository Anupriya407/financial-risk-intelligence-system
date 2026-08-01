from pydantic import BaseModel, ConfigDict


class BaseSchema(BaseModel):
    """Base class for all API schemas."""

    model_config = ConfigDict(
        from_attributes=True,
        extra="forbid",
    )
