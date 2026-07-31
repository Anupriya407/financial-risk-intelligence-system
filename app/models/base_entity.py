from app.models.base import Base
from app.models.mixins.id import IDMixin
from app.models.mixins.timestamp import TimestampMixin


class BaseEntity(Base, IDMixin, TimestampMixin):
    """Base entity for all ORM models."""

    __abstract__ = True
