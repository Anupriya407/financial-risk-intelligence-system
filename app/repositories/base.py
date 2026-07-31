from typing import Generic, TypeVar

from sqlalchemy import func, select
from sqlalchemy.orm import Session

ModelType = TypeVar("ModelType")


class BaseRepository(Generic[ModelType]):
    """Base repository providing common database functionality."""

    def __init__(
        self,
        db: Session,
        model: type[ModelType],
    ) -> None:
        self.db = db
        self.model = model

    def create(
        self,
        obj: ModelType,
    ) -> ModelType:
        """Create a new database record."""

        self.db.add(obj)
        self.db.commit()
        self.db.refresh(obj)

        return obj

    def get_by_id(
        self,
        id: int,
    ) -> ModelType | None:
        """Retrieve a record by its primary key."""

        return self.db.get(self.model, id)

    def get_all(self) -> list[ModelType]:
        """Retrieve all records."""

        statement = select(self.model)

        return list(self.db.scalars(statement).all())

    def update(
        self,
        obj: ModelType,
    ) -> ModelType:
        """Update an existing database record."""

        self.db.commit()
        self.db.refresh(obj)

        return obj

    def delete(
        self,
        obj: ModelType,
    ) -> None:
        """Delete a database record."""

        self.db.delete(obj)
        self.db.commit()

    def exists(
        self,
        id: int,
    ) -> bool:
        """Check whether a record exists."""

        statement = select(self.model).where(self.model.id == id)

        return self.db.scalars(statement).first() is not None

    def count(self) -> int:
        """Return the total number of records."""

        statement = select(func.count()).select_from(self.model)

        return self.db.scalar(statement) or 0
