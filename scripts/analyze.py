from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys


try:
    from ollama import Client, ResponseError  # type: ignore
except Exception as e:  # pragma: no cover
    print("[ERR] Не найдена библиотека 'ollama'. Установи: pip install ollama", file=sys.stderr)
    raise


SYSTEM_PROMPT = (
    "Вы — AI-проверяющий для клинических назначений. Вы НЕ ставите диагноз и не даёте медицинских указаний пациентам. "
    "Ваша задача: оценить корректность диагноза и терапии по кейсу врача, сформировать JSON по схеме. "
    "Если нет базы знаний — массив citations пустой. Все ответы предназначены только для врача и не являются медицинским назначением."
)

# JSON Schema для структурированных ответов (Ollama Structured Outputs)
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "integer", "minimum": 1, "maximum": 100},
        "subscores": {
            "type": "object",
            "properties": {
                "diagnosis": {"type": "integer", "minimum": 0, "maximum": 100},
                "therapy": {"type": "integer", "minimum": 0, "maximum": 100},
                "dosage": {"type": "integer", "minimum": 0, "maximum": 100},
                "interactions": {"type": "integer", "minimum": 0, "maximum": 100},
                "evidence": {"type": "integer", "minimum": 0, "maximum": 100},
            },
            "required": ["diagnosis", "therapy", "dosage", "interactions", "evidence"],
        },
        "critical_errors": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "severity": {"type": "string", "enum": ["RED", "AMBER"]},
                    "message": {"type": "string"},
                    "why": {"type": "string"},
                },
                "required": ["severity", "message"],
            },
        },
        "recommendations": {"type": "array", "items": {"type": "string"}},
        "citations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "section": {"type": "string"},
                    "page": {"type": "string"},
                },
                "required": ["source"],
            },
        },
    },
    "required": ["score", "subscores", "critical_errors", "recommendations"],
}


def build_user_prompt(case_text: str) -> str:
    return (
        "Ниже клинический кейс врача. Оцените корректность и верните ТОЛЬКО JSON, соответствующий схеме.\n"
        "Кейс:\n<<<\n" + case_text + "\n>>>\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Анализ клинического кейса через Ollama")
    parser.add_argument("case_file", type=Path, help="Путь к текстовому файлу кейса (.txt)")
    parser.add_argument("--model", default="llama3.1:8b", help="ID модели Ollama (например, llama3.1:8b или llama3.1:70b)")
    parser.add_argument("--host", default="http://ollama:11434", help="База URL сервера Ollama")
    parser.add_argument("--num_ctx", type=int, default=8192, help="Контекст (токены)")
    parser.add_argument("--temperature", type=float, default=0.0, help="Температура генерации")
    parser.add_argument("--device", default=None, choices=["auto","cpu","cuda"], help="Force device")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 on CUDA")

    args = parser.parse_args()

    if not args.case_file.exists():
        print(f"[ERR] Файл не найден: {args.case_file}", file=sys.stderr)
        return 1

    case_text = args.case_file.read_text(encoding="utf-8")

    client = Client(host=args.host)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(case_text)},
    ]

    try:
        resp = client.chat(
            model=args.model,
            messages=messages,
            format=OUTPUT_SCHEMA,  # строгий JSON по схеме
            stream=False,
            options={"temperature": args.temperature, "num_ctx": args.num_ctx},
        )
    except ResponseError as e:  # noqa: F841
        print("[ERR] Ошибка запроса к Ollama. Частые причины:")
        print(" - Ollama не запущена (запусти приложение или сервис)")
        print(" - Модель не скачана: ollama pull", args.model)
        print("Детали:", e, file=sys.stderr)
        return 2
    except Exception as e:  # pragma: no cover
        print("[ERR] Непредвиденная ошибка:", e, file=sys.stderr)
        return 3

    # По спецификации Structured Outputs: контент приходит как строка с JSON → распарсим
    content = resp.get("message", {}).get("content", "")
    if not content:
        print("[ERR] Пустой ответ от модели", file=sys.stderr)
        return 4

    try:
        result_obj = json.loads(content)
    except json.JSONDecodeError:
        # Случай, если модель вернула нестрогий JSON (редко). Сохраним «как есть» для отладки.
        print("[WARN] Не удалось распарсить JSON строго. Сохраняю сырой ответ.")
        result_obj = {"raw": content}

    out_dir = (args.case_file.parent / ".." / "out").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / (args.case_file.name + ".result.json")

    out_file.write_text(json.dumps(result_obj, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Готово. Результат:", out_file)
    print(json.dumps(result_obj, ensure_ascii=False, indent=2))
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
