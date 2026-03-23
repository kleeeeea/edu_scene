from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from yaml import YAMLError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize scene YAML files into a dataframe and CSV."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing scene*.yaml files.",
    )
    parser.add_argument(
        "--pattern",
        default="scene*.yaml",
        help="Glob pattern for scene config files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "scene_summary.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--error-output",
        type=Path,
        default=Path(__file__).resolve().parent / "scene_summary_errors.csv",
        help="Output CSV path for files that could not be parsed.",
    )
    return parser.parse_args()


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not contain a YAML mapping at the top level.")
    return data


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def pretty_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def prompt_to_text(prompt_items: Any) -> str:
    if not isinstance(prompt_items, list):
        return ""

    lines: list[str] = []
    for item in prompt_items:
        if not isinstance(item, dict):
            continue
        role = item.get("role", "")
        content = item.get("content", "")
        if isinstance(content, str):
            lines.append(f"[{role}] {content}".strip())
        else:
            lines.append(f"[{role}] {compact_json(content)}".strip())
    return "\n".join(lines)


def extract_questions(tasks_content: Any) -> list[str]:
    if not isinstance(tasks_content, list):
        return []

    questions: list[str] = []
    for item in tasks_content:
        if isinstance(item, dict):
            question = item.get("question")
            if question is not None:
                questions.append(str(question))
            else:
                questions.append(compact_json(item))
        else:
            questions.append(str(item))
    return questions


def extract_rubric_fields(evaluation_format: Any) -> list[str]:
    if not isinstance(evaluation_format, list):
        return []

    fields: list[str] = []
    for item in evaluation_format:
        if isinstance(item, dict) and item.get("field") is not None:
            fields.append(str(item["field"]))
    return fields


def build_row(path: Path) -> dict[str, Any]:
    data = read_yaml(path)

    test_model_agent = ((data.get("agents") or {}).get("test_model") or {})
    evaluation = data.get("evaluation") or {}
    tasks = data.get("tasks") or {}

    agent_prompt = test_model_agent.get("prompt") or []
    tasks_content = tasks.get("content") or []
    evaluation_prompt = evaluation.get("prompt") or []
    evaluation_format = evaluation.get("format") or []

    questions = extract_questions(tasks_content)
    rubric_fields = extract_rubric_fields(evaluation_format)

    return {
        "scene_file": path.name,
        "scene_stem": path.stem,
        "task_mode": tasks.get("mode"),
        "task_count": len(questions),
        "test_model_ref": test_model_agent.get("model"),
        "test_model_prompt_json": pretty_json(agent_prompt),
        "test_model_prompt_text": prompt_to_text(agent_prompt),
        "tasks_content_json": pretty_json(tasks_content),
        "sample_questions_json": pretty_json(questions),
        "sample_questions_preview": "\n\n".join(questions[:3]),
        "evaluation_name": evaluation.get("name"),
        "evaluation_model_ref": evaluation.get("model"),
        "evaluation_prompt_json": pretty_json(evaluation_prompt),
        "evaluation_prompt_text": prompt_to_text(evaluation_prompt),
        "evaluation_format_json": pretty_json(evaluation_format),
        "rubric_fields_json": pretty_json(rubric_fields),
        "rubric_field_count": len(rubric_fields),
        "rubric_fields_preview": ", ".join(rubric_fields[:10]),
        "format_mode": evaluation.get("format_mode", data.get("format_mode")),
    }


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_path = args.output.resolve()
    error_output_path = args.error_output.resolve()

    yaml_paths = sorted(input_dir.glob(args.pattern))
    if not yaml_paths:
        raise FileNotFoundError(
            f"No files matched pattern {args.pattern!r} in {input_dir}"
        )

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for path in yaml_paths:
        try:
            rows.append(build_row(path))
        except (ValueError, YAMLError) as exc:
            errors.append(
                {
                    "scene_file": path.name,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )

    if not rows:
        raise RuntimeError("No valid YAML files were parsed successfully.")

    dataframe = pd.DataFrame(rows).sort_values("scene_file").reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(errors).to_csv(error_output_path, index=False, encoding="utf-8-sig")

    print(f"Loaded {len(dataframe)} scene files from {input_dir}")
    print(f"Wrote summary CSV to {output_path}")
    if errors:
        print(f"Skipped {len(errors)} invalid YAML files")
        print(f"Wrote parse errors to {error_output_path}")
    print(dataframe[["scene_file", "task_count", "rubric_field_count"]].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
