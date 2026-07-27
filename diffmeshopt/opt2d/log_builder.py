import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


def get_git_branch_and_commit(repo_root: str | Path | None = None) -> tuple[str, str]:
    """Gets the current git branch and short commit hash of the repository.

    Format is: "<branch_name> @ <short_sha>"
    """
    if repo_root is None:
        repo_root = Path.cwd()

    for candidate in [Path(repo_root), *Path(repo_root).parents]:
        if (candidate / ".git").exists():
            repo_root = candidate
            break

    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(repo_root),
            text=True,
        ).strip()
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo_root),
            text=True,
        ).strip()
    except Exception:
        branch, sha = "unknown", "unknown"

    return f"{branch} @ {sha}"


def _enum_or_value(value: Any) -> Any:
    return value.value if isinstance(value, Enum) else value


def _get_latest_metrics(trainer: Any) -> dict[str, Any]:
    trainer_obj = getattr(trainer, "trainer", None)
    if trainer_obj is None:
        return {}

    loggers = getattr(trainer_obj, "loggers", [])
    logger = next(
        (l for l in loggers if hasattr(l, "history")),
        None,
    )
    if logger is None:
        return {}

    history = getattr(logger, "history", None) or []
    if not history:
        return {}

    last_entry = history[-1]
    if hasattr(last_entry, "as_dict"):
        last_entry = last_entry.as_dict()

    if isinstance(last_entry, dict):
        return last_entry

    return {}


@dataclass
class ExperimentLogEntry:
    run_id: str
    date: str | None = None
    author: str = "<name>"
    branch_commit: str = get_git_branch_and_commit()
    notebook_script: str = "<path>"
    device: str = "<cpu|cuda|mps>"
    seeds: str = "<e.g., 1234, 145>"
    status: str = "complete"

    goal: str = ""
    dataset_path: str = ""
    sample_key: str = ""
    preprocessing: str = ""
    frame_mapping_metadata: dict[str, Any] = field(default_factory=dict)

    config: dict[str, Any] = field(default_factory=dict)
    regularization_weights: dict[str, Any] = field(default_factory=dict)
    runtime_notes: dict[str, Any] = field(default_factory=dict)
    quantitative_results: dict[str, Any] = field(default_factory=dict)
    template_diagnostics: dict[str, Any] = field(default_factory=dict)
    visual_checks: dict[str, Any] = field(default_factory=dict)
    interpretation: list[str] = field(default_factory=list)
    next_action: list[str] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def _fmt(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, int):
            return f"{value:d}"
        if isinstance(value, float):
            return f"{value:.6g}"
        if isinstance(value, (list, tuple)):
            return ", ".join(str(v) for v in value)
        if isinstance(value, dict):
            return "; ".join(f"{k}={v}" for k, v in value.items())
        return str(value)

    def _table(self, rows: list[tuple[str, Any]]) -> str:
        if not rows:
            return ""
        lines = ["| Field | Value |", "| --- | --- |"]
        for key, value in rows:
            if value in (None, ""):
                continue
            lines.append(f"| {key} | {self._fmt(value)} |")
        return "\n".join(lines)

    def _kv_grid_table(
        self,
        rows: list[tuple[str, Any]],
        *,
        pairs_per_row: int = 3,  # 3 pairs => 6 columns
    ) -> str:
        """Compact multi-column key/value markdown table."""
        filtered = [(k, v) for k, v in rows if v not in (None, "")]
        if not filtered:
            return ""

        pairs_per_row = max(1, int(pairs_per_row))
        ncols = pairs_per_row * 2

        header_cells: list[str] = []
        align_cells: list[str] = []
        for _ in range(pairs_per_row):
            header_cells.extend(["Field", "Value"])
            align_cells.extend(["---", "---"])

        lines = [
            "| " + " | ".join(header_cells) + " |",
            "| " + " | ".join(align_cells) + " |",
        ]

        i = 0
        while i < len(filtered):
            chunk = filtered[i : i + pairs_per_row]
            row_cells: list[str] = []
            for k, v in chunk:
                row_cells.extend([str(k), self._fmt(v)])
            # pad final row
            while len(row_cells) < ncols:
                row_cells.extend(["", ""])
            lines.append("| " + " | ".join(row_cells) + " |")
            i += pairs_per_row

        return "\n".join(lines)

    def _list(self, items: list[str]) -> str:
        if not items:
            return ""
        return "\n".join(f"- {item}" for item in items)

    def to_markdown(self) -> str:
        lines: list[str] = []

        lines.append("## Run Summary")
        lines.append("")
        lines.append(
            self._table([
                ("Run ID", self.run_id),
                ("Author", self.author),
                ("Branch / Commit", self.branch_commit),
                ("Notebook / Script", self.notebook_script),
                ("Device", self.device),
                ("Seed(s)", self.seeds),
                ("Status", self.status),
            ])
        )
        lines.append("")

        lines.append("### 1) Goal")
        lines.append(f"- {self.goal}" if self.goal else "- <goal>")
        lines.append("")

        lines.append("### 2) Data")
        lines.append(
            self._kv_grid_table(
                [
                    ("Dataset path", self.dataset_path),
                    ("Slice / sample key", self.sample_key),
                    ("Preprocessing", self.preprocessing),
                ],
                pairs_per_row=2,  # 4 columns
            )
        )
        if self.frame_mapping_metadata:
            lines.append("")
            lines.append(
                self._kv_grid_table(
                    list(self.frame_mapping_metadata.items()),
                    pairs_per_row=3,  # 6 columns
                )
            )
        lines.append("")

        lines.append("### 3) Configuration")
        config_rows = [
            ("Refiner", self.config.get("refiner", "")),
            ("Template", self.config.get("template", "")),
            ("Data loss type", self.config.get("data_loss_type", "")),
            ("Learning rate", self.config.get("learning_rate", "")),
            ("Num cp", self.config.get("num_cp", "")),
            ("RBF sigma", self.config.get("rbf_sigma", "")),
            ("Profile length", self.config.get("profile_length", "")),
            ("Profile width", self.config.get("profile_width", "")),
            ("Shape loss weight", self.config.get("shape_loss_weight", "")),
            ("Min peak ratio", self.config.get("min_peak_ratio", "")),
        ]
        lines.append(self._kv_grid_table(config_rows, pairs_per_row=2))  # 4 columns
        if self.regularization_weights:
            lines.append("")
            lines.append("**Regularization weights**")
            lines.append("")
            lines.append(
                self._kv_grid_table(
                    list(self.regularization_weights.items()),
                    pairs_per_row=3,  # 6 columns
                )
            )
        lines.append("")

        lines.append("### 4) Runtime Notes")
        if self.runtime_notes:
            lines.append(self._kv_grid_table(list(self.runtime_notes.items()), pairs_per_row=2))
        else:
            lines.append("- <none>")
        lines.append("")

        lines.append("### 5) Quantitative Results")
        lines.append(
            self._kv_grid_table(
                list(self.quantitative_results.items()),
                pairs_per_row=3,  # 6 columns
            )
        )
        lines.append("")

        lines.append("### 6) Template Diagnostics")
        if self.template_diagnostics:
            lines.append(
                self._kv_grid_table(
                    list(self.template_diagnostics.items()),
                    pairs_per_row=3,  # 6 columns
                )
            )
        else:
            lines.append("- <none>")
        lines.append("")

        lines.append("### 7) Visual Checks")
        if self.visual_checks:
            lines.append(self._kv_grid_table(list(self.visual_checks.items()), pairs_per_row=2))
        else:
            lines.append("- <none>")
        lines.append("")

        lines.append("### 8) Interpretation")
        interpretation = self._list(self.interpretation)
        lines.append(interpretation or "- <interpretation>")
        lines.append("")

        lines.append("### 9) Next Action")
        next_action = self._list(self.next_action)
        lines.append(next_action or "- <next action>")
        lines.append("")

        lines.append("### 10) Artifacts")
        if self.artifacts:
            lines.append(self._kv_grid_table(list(self.artifacts.items()), pairs_per_row=2))
        else:
            lines.append("- <none>")

        return "\n".join(lines)


def append_experiment_log(path: str | Path, entry: ExperimentLogEntry) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    marker = f"<!-- run_id: {entry.run_id} -->"
    block = f"{marker}\n{entry.to_markdown()}"

    if path.exists():
        existing = path.read_text()
        if marker in existing:
            start = existing.index(marker)
            next_marker = existing.find("<!-- run_id:", start + len(marker))
            if next_marker == -1:
                end = len(existing)
            else:
                end = next_marker
            content = existing[:start] + block + existing[end:]
        else:
            content = existing.rstrip() + "\n\n---\n\n" + block
    else:
        content = "# Experiment Log\n\n" + block

    path.write_text(content.rstrip() + "\n")


def create_experiment_log_entry_from_trainer(
    trainer: Any,
    run_id: str,
    *,
    extra_metrics: dict[str, float] | None = None,
    initial_metrics: dict[str, float] | None = None,
    template_diagnostics: dict[str, float] | None = None,
    author: str = "Saugat Kandel",
    notebook_script: str = "<path>",
    dataset_path: str = "",
    sample_key: str = "",
    preprocessing: str = "",
    goal: str = "",
    device: str | None = None,
    seeds: str = "145",
    status: str = "complete",
    frame_mapping_metadata: dict[str, Any] | None = None,
) -> "ExperimentLogEntry":
    refiner = getattr(trainer, "refiner", None)
    trainer_obj = getattr(trainer, "trainer", None)
    output_dir = getattr(trainer, "output_dir", None)

    try:
        refiner = trainer.refiner
        props = refiner.props
        template_model = refiner.template_model
        template_props = template_model.props
    except AttributeError as exc:
        raise AttributeError(
            "Experiment log requires trainer.refiner.props and "
            "trainer.refiner.template_model.props"
        ) from exc

    latest_metrics = _get_latest_metrics(trainer)
    if extra_metrics:
        latest_metrics.update(extra_metrics)

    config: dict[str, Any] = {}
    if props is not None:
        config["refiner"] = _enum_or_value(getattr(refiner, "refiner_kind", None) or "")
        config["learning_rate"] = getattr(props, "learning_rate", None)
        config["shape_loss_weight"] = getattr(props, "shape_loss_weight", None)
        config["num_cp"] = getattr(props, "rbf_num_control_points", None) or getattr(
            props, "contour_num_control_points", None
        )
        config["rbf_sigma"] = getattr(props, "rbf_kernel_sigma", None)
        config["profile_length"] = getattr(props, "profile_length", None)
        config["profile_width"] = getattr(props, "profile_width", None)
        config["data_loss_type"] = _enum_or_value(getattr(props, "data_loss_type", None))

    if template_props is not None:
        config["template"] = _enum_or_value(getattr(template_model, "template_kind", None) or "")
        config["min_peak_ratio"] = getattr(template_props, "min_peak_ratio", None)

    quantitative_results: dict[str, Any] = {}
    for key in ("mean_dist", "hausdorff_dist", "p95_dist", "total_loss", "data_loss"):
        if key in latest_metrics:
            quantitative_results[key] = latest_metrics[key]

    if initial_metrics:
        for key, value in initial_metrics.items():
            quantitative_results[f"initial_{key}"] = value

    runtime_notes: dict[str, Any] = {}
    if trainer_obj is not None:
        runtime_notes["global_step"] = getattr(trainer_obj, "global_step", None)

    regularization_weights: dict[str, Any] = {}
    if props is not None:
        reg_weights = getattr(props, "initial_regularization_weights", None)
        if isinstance(reg_weights, dict):
            regularization_weights.update(reg_weights)

    artifacts: dict[str, Any] = {}
    if output_dir is not None:
        artifacts["output_dir"] = str(output_dir)
        artifacts["final_state"] = str(Path(output_dir) / "final.pkl")

    return ExperimentLogEntry(
        run_id=run_id,
        author=author,
        notebook_script=notebook_script,
        device=device or "cpu",
        seeds=seeds,
        status=status,
        goal=goal,
        dataset_path=dataset_path,
        sample_key=sample_key,
        preprocessing=preprocessing,
        frame_mapping_metadata=frame_mapping_metadata or {},
        config=config,
        regularization_weights=regularization_weights,
        runtime_notes=runtime_notes,
        quantitative_results=quantitative_results,
        template_diagnostics=template_diagnostics or {},
        interpretation=[],
        next_action=[],
        artifacts=artifacts,
    )
