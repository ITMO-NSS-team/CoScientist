"""Deterministic renderer: StructuredTZ -> the human-readable ТЗ document.

Reproduces the structure of the reference document
``Пример_уточненного_структурированного_ТЗ_для_агентов.md``:

  1. заголовок (версия, дата, статус документа)
  2. правила интерпретации полей (легенда статусов)
  3. исходный запрос заказчика в свободной форме
  4. единая структура собранных данных (обзорная таблица по блокам)
  5. таблицы блоков |Поле|Значение|Статус|
  6. поля, которые остаются свободными или незаполненными
  7. проверка соответствия исходному опроснику

The renderer is pure formatting: everything it prints comes from the
validated StructuredTZ (or is computed from it), so the document can never
disagree with the machine-readable ТЗ the rest of the pipeline consumes.
"""
from __future__ import annotations

import json
from datetime import date
from typing import Any, Union

from CoScientist.microfluidics.models import (
    CANONICAL_BLOCKS,
    OPEN_STATUSES,
    StructuredTZ,
)

_LEGEND = """\
| Статус поля            | Как интерпретировать                                                                                                        |
| ---------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| Задано заказчиком      | Конкретное значение получено от заказчика и может использоваться агентами как входное ограничение или критерий               |
| Уточнено оператором    | Значение добавлено оператором/агентом постановки ТЗ из контекста и может использоваться как рабочее ограничение              |
| Не задано              | Значение отсутствует; агент не должен его придумывать, но может запросить уточнение или выполнить поиск вариантов            |
| Свободный комментарий  | Неформализованная информация; не используется как жёсткий критерий ранжирования, пока не переведена в конкретный параметр    |
| Рассчитывается агентом | Значение должно быть определено на следующем этапе работы системы                                                            |"""

_OPEN_GUIDANCE = {
    "не задано": "Не подменять предположениями; запросить уточнение или выполнить поиск вариантов",
    "рассчитывается агентом": "Определяется на следующем этапе работы системы",
}


def _cap(status: str) -> str:
    return status[:1].upper() + status[1:] if status else status


def _cell(text: str) -> str:
    """Make arbitrary text safe inside a Markdown table cell."""
    return str(text).replace("|", "\\|").replace("\n", " ").strip()


def _coerce(tz: Union[StructuredTZ, dict, str, Any]) -> StructuredTZ:
    if isinstance(tz, StructuredTZ):
        return tz
    if isinstance(tz, str):
        tz = json.loads(tz)
    return StructuredTZ.model_validate(tz)


def render_tz_document(tz: Union[StructuredTZ, dict, str], version: str = "v1") -> str:
    """Render the full ТЗ Markdown document from a (validated) StructuredTZ."""
    tz = _coerce(tz)
    out: list[str] = []

    # 1 ── header
    out.append("# Уточнённое структурированное ТЗ для системы ИИ-агентов")
    out.append("")
    out.append(f"Версия: {version}  ")
    out.append(f"Дата: {date.today().strftime('%d.%m.%Y')}  ")
    out.append(
        "Статус документа: сформирован агентом постановки ТЗ (CoScientist, "
        "кейс «микрофлюидика»)"
    )
    out.append("\n---\n")

    # 2 ── interpretation rules
    out.append("## Правила интерпретации полей\n")
    out.append(
        "Документ является единым структурированным входом для последующих "
        "ИИ-агентов. Для каждого поля указывается статус:\n"
    )
    out.append(_LEGEND)
    out.append(
        "\nНеконкретные формулировки («доступное сырьё», «устойчивые поставки», "
        "«приемлемая цена») не используются как жёсткие критерии: для "
        "автоматической обработки они переведены в измеримые поля или помечены "
        "как свободный комментарий."
    )
    out.append("\n---\n")

    # 3 ── original request
    out.append("## Исходный запрос заказчика в свободной форме\n")
    request = (tz.original_request or "").strip() or "Не задан"
    out.append("\n".join(f"> {line}" for line in request.splitlines() if line.strip()))
    out.append("\n---\n")

    # 4 ── overview table
    out.append("## Единая структура собранных данных\n")
    out.append("| Блок данных | Собрано в ТЗ | Заполненность | Использование далее |")
    out.append("| --- | --- | --- | --- |")
    for block in tz.blocks:
        total = len(block.fields)
        filled = sum(1 for f in block.fields if f.is_set())
        if total == 0 or filled == 0:
            collected = "Нет"
        elif filled == total:
            collected = "Да"
        else:
            collected = "Частично"
        usage = block.usage or "—"
        out.append(
            f"| {_cell(block.title)} | {collected} | {filled} из {total} полей "
            f"| {_cell(usage)} |"
        )
    out.append("\n---\n")

    # 5 ── per-block tables
    for block in tz.blocks:
        out.append(f"## {block.title.strip()}\n")
        out.append("| Поле | Значение | Статус |")
        out.append("| --- | --- | --- |")
        for f in block.fields:
            out.append(f"| {_cell(f.name)} | {_cell(f.value)} | {_cap(f.status)} |")
        out.append("\n---\n")

    # 6 ── open fields
    open_rows = [
        (block.title, f)
        for block in tz.blocks
        for f in block.fields
        if f.status in OPEN_STATUSES
    ]
    out.append("## Поля, которые остаются свободными или незаполненными\n")
    if open_rows:
        out.append(
            "Эти поля не должны подменяться предположениями агентов без "
            "отдельной пометки.\n"
        )
        out.append("| Блок | Поле | Текущее значение | Как использовать дальше |")
        out.append("| --- | --- | --- | --- |")
        for title, f in open_rows:
            out.append(
                f"| {_cell(title)} | {_cell(f.name)} | {_cell(f.value)} "
                f"| {_OPEN_GUIDANCE[f.status]} |"
            )
    else:
        out.append("Все поля ТЗ заполнены.")
    out.append("\n---\n")

    # 7 ── questionnaire coverage check
    out.append("## Проверка соответствия исходному опроснику\n")
    out.append("| Элемент опросника | Наличие в ТЗ | Комментарий |")
    out.append("| --- | --- | --- |")
    for title in CANONICAL_BLOCKS:
        block = tz.block(title)
        if block is None or not block.fields:
            out.append(f"| {title} | Отсутствует | Блок не сформирован |")
            continue
        filled = sum(1 for f in block.fields if f.is_set())
        out.append(
            f"| {title} | Есть | Заполнено {filled} из {len(block.fields)} полей |"
        )
    extra = [b.title for b in tz.blocks if b.title not in CANONICAL_BLOCKS]
    if extra:
        for title in extra:
            out.append(f"| {_cell(title)} | Дополнительный блок | Вне опросника |")

    return "\n".join(out).rstrip() + "\n"


__all__ = ["render_tz_document"]
