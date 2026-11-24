from collections.abc import Iterator
from datetime import date, timedelta
from random import Random

from verifine.data.schemas import Case, Language, Record, Task


def optional(rng: Random, value: str) -> str | None:
    return value if rng.random() >= 0.24 else None


def invoice(rng: Random, lang: Language, marker: str) -> tuple[Record, list[str]]:
    ref = f"INV-{marker}"
    vendor = optional(rng, f"Vetra {rng.choice(('Axis', 'Lumen', 'Noma', 'Tilia'))}{marker[-3:]}")
    total = rng.randrange(2100, 255_000)
    currency = rng.choice(("UAH", "USD", "EUR"))
    due = optional(rng, (date(2025, 1, 1) + timedelta(days=rng.randrange(900))).isoformat())
    uk = lang == "uk"
    amount = f"{total // 100}{',' if uk else '.'}{total % 100:02d} {currency}"
    facts = [
        f"Рахунок № {ref}" if uk else f"Invoice reference {ref}",
        "Замовник: Orion Paper" if uk else "Billed to: Orion Paper",
        (f"Постачальник: {vendor}" if uk else f"Supplier: {vendor}") if vendor else "",
        rng.choice((f"До сплати: {amount}", f"Підсумкова сума: {amount}"))
        if uk
        else rng.choice((f"Final amount payable: {amount}", f"Total due: {amount}")),
        (f"Строк оплати: {due}" if uk else f"Pay by {due}") if due else "",
    ]
    return {
        "invoice_id": ref,
        "supplier": vendor,
        "total_minor": total,
        "currency": currency,
        "due_date": due,
    }, facts


def clinical(rng: Random, lang: Language, marker: str) -> tuple[Record, list[str]]:
    visit = optional(rng, (date(2024, 1, 1) + timedelta(days=rng.randrange(1050))).isoformat())
    uk = lang == "uk"
    symptoms = (
        ("сухий кашель", "запаморочення", "втома", "біль у зап'ясті")
        if uk
        else (
            "dry cough",
            "dizziness",
            "fatigue",
            "wrist pain",
        )
    )
    findings = (
        ("набряк відсутній", "чутливість збережена", "дихання рівномірне")
        if uk
        else (
            "no swelling",
            "sensation intact",
            "regular breathing",
        )
    )
    symptom, finding = optional(rng, rng.choice(symptoms)), optional(rng, rng.choice(findings))
    follow = rng.choice((None, 3, 5, 7, 10, 14, 21, 28))
    facts = [
        f"Навчальний запис DEMO-{marker}" if uk else f"Fictional record DEMO-{marker}",
        (f"Візит: {visit}" if uk else f"Visit date: {visit}") if visit else "",
        (f"Скарга: {symptom}" if uk else f"Reported symptom: {symptom}") if symptom else "",
        (f"Огляд: {finding}" if uk else f"Documented finding: {finding}") if finding else "",
        (f"Повторний візит через {follow} днів" if uk else f"Follow-up in {follow} days")
        if follow is not None
        else "",
    ]
    return {
        "visit_date": visit,
        "symptom": symptom,
        "documented_finding": finding,
        "follow_up_days": follow,
    }, facts


def contract(rng: Random, lang: Language, marker: str) -> tuple[Record, list[str]]:
    ref = f"AGR-{marker}"
    counterparty = optional(rng, f"Ravel {rng.choice(('Labs', 'Trade', 'Works'))}{marker[-3:]}")
    effective = optional(rng, (date(2025, 2, 1) + timedelta(days=rng.randrange(660))).isoformat())
    uk = lang == "uk"
    notice = rng.choice((None, 15, 30, 45, 60, 90))
    jurisdiction = optional(
        rng,
        rng.choice(("Україна", "Польща", "Естонія") if uk else ("Ukraine", "Poland", "Estonia")),
    )
    facts = [
        f"Угода {ref}" if uk else f"Agreement {ref}",
        (f"Контрагент: {counterparty}" if uk else f"Counterparty: {counterparty}")
        if counterparty
        else "",
        (f"Набирає чинності {effective}" if uk else f"Effective from {effective}")
        if effective
        else "",
        (f"Строк повідомлення: {notice} днів" if uk else f"Give {notice} days' notice")
        if notice is not None
        else "",
        (f"Застосовне право: {jurisdiction}" if uk else f"Governing law: {jurisdiction}")
        if jurisdiction
        else "",
    ]
    return {
        "contract_id": ref,
        "counterparty": counterparty,
        "effective_date": effective,
        "notice_days": notice,
        "jurisdiction": jurisdiction,
    }, facts


def support(rng: Random, lang: Language, marker: str) -> tuple[Record, list[str]]:
    tool = rng.choice(("assign_priority", "lookup_ticket", None))
    uk = lang == "uk"
    ref = optional(rng, f"SUP-{marker}") if tool else None
    priority = rng.choice(("low", "normal", "high", None)) if tool == "assign_priority" else None
    localized = dict(
        zip(("low", "normal", "high"), ("низький", "звичайний", "високий"), strict=True)
    )
    if tool == "assign_priority":
        request = (
            rng.choice(("Встановити пріоритет", "Змінити пріоритет", "Зміна пріоритету"))
            if uk
            else rng.choice(("Change priority", "Set ticket priority", "Priority update"))
        )
    elif tool == "lookup_ticket":
        request = (
            rng.choice(("Статус заявки", "Перевірити стан", "Перевірка статусу"))
            if uk
            else rng.choice(("Ticket status", "Check the status", "Status lookup"))
        )
    else:
        request = (
            "Дякую за попередню відповідь. Запитів немає."
            if uk
            else "Thank you for the earlier reply. No request today."
        )
    priority_label = localized.get(priority, priority) if uk and priority else priority
    facts = [
        request,
        f"Квиток: {ref}" if uk and ref else ref or "",
        f"→ {priority_label}" if priority_label else "",
        f"Канал: WEB-{marker}" if uk else f"Channel: WEB-{marker}",
    ]
    return {"tool": tool, "arguments": {"ticket_id": ref, "priority": priority}}, facts


BUILDERS = {"invoice": invoice, "clinical": clinical, "contract": contract, "support": support}


def render(
    rng: Random,
    facts: list[str],
    lang: Language,
    split: str,
    profile: int,
    marker: str,
) -> str:
    facts = [text for text in facts if text]
    uk = lang == "uk"
    if split == "train":
        prefixes = (
            ("Вхідна форма", "Чернетка з форми", "Заявка після імпорту")
            if uk
            else (
                "Inbound form",
                "Draft from form",
                "Imported request",
            )
        )
        suffix = "" if profile % 2 else ("\n— Кінець форми" if uk else "\n— End of form")
        return prefixes[profile % 3] + "\n" + "\n".join(f"- {fact}" for fact in facts) + suffix
    if split == "validation":
        header = "Зведення відділу | Стислий запис" if uk else "Team digest / short record"
        rng.shuffle(facts)
        return header + "\n" + " | ".join(facts)
    introductions = (
        (
            "Витяг з переписки. Архів підрозділу передав матеріал без вкладень.",
            "Оцифрований фрагмент. Позначки між рядками додала система імпорту.",
            "Робоча копія для перевірки якості. Дані відтворено з основної частини запису.",
        )
        if uk
        else (
            "Excerpt from correspondence. The archive forwarded the body without attachments.",
            "Digitized excerpt. Formatting and line breaks were added by the intake system.",
            "Review copy for quality checks. Details below were recovered from the main body.",
        )
    )
    punctuation = ("\n  • ", "; ", "\n  ↳ ")[profile % 3]
    rng.shuffle(facts)
    origin = rng.choice(("Atlas", "Beryl", "Coral", "Delta", "Ember", "Flint", "Grove", "Halo"))
    batch = rng.choice(("Birch", "Cedar", "Lime", "Maple", "Oak", "Pine", "Spruce", "Willow"))
    archive = (
        f"Архів {origin}, пакет {batch}-{marker}"
        if uk
        else f"Archive {origin}, intake batch {batch}-{marker}"
    )
    return archive + "\n" + introductions[profile % 3] + "\n" + punctuation.join(facts)


def synthetic_cases(seed: int) -> Iterator[tuple[Case, str]]:
    rng = Random(seed)
    tasks: tuple[Task, ...] = ("invoice", "clinical", "contract", "support")
    for split, count, prefix in (
        ("train", 800, "TR"),
        ("validation", 120, "VA"),
        ("golden", 300, "GO"),
    ):
        for index in range(count):
            task = tasks[(index // 2) % len(tasks)]
            lang: Language = "uk" if index % 2 == 0 else "en"
            profile = (index // 8) % 6
            marker = f"{prefix}{index:04}"
            expected, facts = BUILDERS[task](rng, lang, marker)
            yield (
                Case(
                    id=f"{task}-{marker.lower()}",
                    group=f"{split}-{task}-{lang}-{profile}",
                    task=task,
                    language=lang,
                    document=render(rng, facts, lang, split, profile, marker),
                    expected=expected,
                    synthetic=True,
                ),
                split,
            )
