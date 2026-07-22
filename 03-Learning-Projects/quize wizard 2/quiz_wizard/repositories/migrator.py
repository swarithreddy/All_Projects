from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from quiz_wizard.config import LEGACY_FILE_MAP
from quiz_wizard.models.question import Question, QuestionBank

logger = logging.getLogger(__name__)

_EXPLANATION_PREFIX = re.compile(r"^Explanation:\s*", re.IGNORECASE)


def parse_legacy_txt(text: str) -> list[Question]:
    """Parse V1 8-line question blocks (same rules as main_operation.read_questions)."""
    lines = text.splitlines(keepends=True)
    questions: list[Question] = []
    i = 0
    while i < len(lines):
        if lines[i].strip().isdigit():
            number = int(lines[i].strip())
            prompt = lines[i + 1].strip()
            options = tuple(
                lines[i + offset][3:].strip() for offset in range(2, 6)
            )  # type: ignore[assignment]
            if len(options) != 4:
                raise ValueError(f"Expected 4 options at question {number}")
            answer_index = int(lines[i + 6].strip())
            explanation_raw = lines[i + 7].strip()
            explanation = _EXPLANATION_PREFIX.sub("", explanation_raw).strip()
            questions.append(
                Question(
                    id=number,
                    prompt=prompt,
                    options=options,  # type: ignore[arg-type]
                    answer_index=answer_index,
                    explanation=explanation,
                )
            )
            i += 8
        else:
            i += 1
    return questions


def bank_to_dict(bank: QuestionBank) -> dict:
    return {
        "category": bank.category,
        "difficulty": bank.difficulty,
        "questions": [
            {
                "id": q.id,
                "prompt": q.prompt,
                "options": list(q.options),
                "answer_index": q.answer_index,
                "explanation": q.explanation,
            }
            for q in bank.questions
        ],
    }


def migrate_file(src: Path, category: str, difficulty: str) -> QuestionBank:
    text = src.read_text(encoding="utf-8")
    questions = parse_legacy_txt(text)
    return QuestionBank(category=category, difficulty=difficulty, questions=questions)


def migrate_legacy_directory(legacy_dir: Path, output_dir: Path) -> list[Path]:
    """Convert all known V1 *.txt banks into JSON under output_dir."""
    written: list[Path] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for stem, (category, difficulty) in LEGACY_FILE_MAP.items():
        src = legacy_dir / f"{stem}.txt"
        if not src.exists():
            logger.warning("Missing legacy file: %s", src)
            continue
        bank = migrate_file(src, category, difficulty)
        category_dir = output_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)
        dest = category_dir / f"{difficulty}.json"
        dest.write_text(
            json.dumps(bank_to_dict(bank), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        written.append(dest)
        logger.info("Migrated %s -> %s (%d questions)", src.name, dest, len(bank))
    return written
