from datetime import date
from typing import Annotated, Literal, Self, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Task: TypeAlias = Literal["invoice", "clinical", "contract", "support"]
Language: TypeAlias = Literal["uk", "en"]
Currency: TypeAlias = Literal["UAH", "EUR", "USD"]
Priority: TypeAlias = Literal["low", "normal", "high"]
Value: TypeAlias = str | int | None
Record: TypeAlias = dict[str, Value | dict[str, Value]]
Text = Annotated[str, Field(min_length=1, max_length=180, pattern=r"\S")]
DateText = Annotated[str, Field(pattern=r"^\d{4}-\d{2}-\d{2}$")]
Money = Annotated[int, Field(ge=0, le=100_000_000_000)]


class StrictModel(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", str_strip_whitespace=True)

    @field_validator("due_date", "visit_date", "effective_date", check_fields=False)
    @classmethod
    def calendar_date(cls, value: str | None) -> str | None:
        if value is not None:
            date.fromisoformat(value)
        return value


class Invoice(StrictModel):
    invoice_id: Text | None
    supplier: Text | None
    total_minor: Money | None
    currency: Currency | None
    due_date: DateText | None


class ClinicalNote(StrictModel):
    visit_date: DateText | None
    symptom: Text | None
    documented_finding: Text | None
    follow_up_days: Annotated[int, Field(ge=0, le=365)] | None


class Contract(StrictModel):
    contract_id: Text | None
    counterparty: Text | None
    effective_date: DateText | None
    notice_days: Annotated[int, Field(ge=0, le=3650)] | None
    jurisdiction: Text | None


class SupportArguments(StrictModel):
    ticket_id: Text | None
    priority: Priority | None


class SupportToolCall(StrictModel):
    tool: Literal["assign_priority", "lookup_ticket"] | None
    arguments: SupportArguments

    @model_validator(mode="after")
    def tool_scope(self) -> "SupportToolCall":
        if self.tool != "assign_priority" and self.arguments.priority is not None:
            raise ValueError("Only assign_priority accepts a priority")
        if self.tool is None and self.arguments.ticket_id is not None:
            raise ValueError("A null tool requires null arguments")
        return self


SCHEMAS: dict[Task, type[StrictModel]] = {
    "invoice": Invoice,
    "clinical": ClinicalNote,
    "contract": Contract,
    "support": SupportToolCall,
}


class Case(StrictModel):
    id: Annotated[str, Field(min_length=1)]
    group: Annotated[str, Field(min_length=1)]
    task: Task
    language: Language
    document: Annotated[str, Field(min_length=1, max_length=16000)]
    expected: Record
    synthetic: bool = False

    @model_validator(mode="after")
    def validate_expected(self) -> Self:
        self.expected = SCHEMAS[self.task].model_validate(self.expected).model_dump()
        return self
