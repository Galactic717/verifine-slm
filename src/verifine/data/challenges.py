from verifine.data.schemas import Case, Task

ARCHIVE_EN: dict[Task, str] = {
    "invoice": (
        "Invoice reference INV-OLD-513; supplier Archival Pine; final amount payable: 74.90 EUR; "
        "pay by 2024-04-17. This invoice was voided."
    ),
    "contract": (
        "Agreement AGR-OLD-833; counterparty Archival Pine; effective from 2024-04-17; give 45 "
        "days' notice; governing law: Portugal. This version was a withdrawn draft."
    ),
    "clinical": (
        "Visit date: 2024-04-17; reported symptom: elbow pain; documented finding: redness; "
        "follow-up in 90 days. This is a previous fictional visit."
    ),
    "support": (
        "Set ticket priority for SUP-OLD-729 to high. The quoted request was already resolved."
    ),
}

ARCHIVE_UK: dict[Task, str] = {
    "invoice": (
        "Рахунок № INV-OLD-513; постачальник Archival Pine; до сплати 74,90 EUR; строк оплати "
        "2024-04-17. Рахунок скасовано."
    ),
    "contract": (
        "Угода AGR-OLD-833; контрагент Archival Pine; набирає чинності 2024-04-17; "
        "строк повідомлення 45 днів; застосовне право: Португалія. Цю чернетку відкликано."
    ),
    "clinical": (
        "Візит: 2024-04-17; скарга: біль у лікті; огляд: почервоніння; повторний візит "
        "через 90 днів. Це попередній вигаданий візит."
    ),
    "support": ("Встановити пріоритет заявки SUP-OLD-729 → високий. Цитований запит уже закрито."),
}


def add_archival_distractor(case: Case, *, appendix: bool) -> Case:
    if not case.synthetic:
        raise ValueError("Archive augmentation requires a synthetic training case")
    if case.language == "uk":
        archive = "Архівна довідка за минулий період\n" + ARCHIVE_UK[case.task]
        current = "Поточний запис для обробки\n" + case.document
    else:
        archive = "Archived material from an earlier case\n" + ARCHIVE_EN[case.task]
        current = "Current record for processing\n" + case.document
    pieces = (current, archive) if appendix else (archive, current)
    return case.model_copy(update={"document": "\n\n".join(pieces)})
