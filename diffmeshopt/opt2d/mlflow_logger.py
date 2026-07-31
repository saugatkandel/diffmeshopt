import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlflow

from diffmeshopt.opt2d.log_builder import (
    ExperimentLogEntry,
    create_experiment_log_entry_from_trainer,
)
from diffmeshopt.opt2d.repo_context import resolve_repo_root


def _load_env_file_if_present() -> None:
    """Load .env from repo root if present (without overriding existing env vars)."""
    env_path = resolve_repo_root(Path(__file__).resolve().parent) / ".env"
    if env_path.exists():
        print(f"Environment file exists in {env_path}. Trying to load MLFLOW_TRACKING_URI...")
    else:
        return

    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


def _ensure_mlflow_configured(
    *,
    tracking_uri: str | None = None,
    experiment_name: str | None = None,
) -> str:
    _load_env_file_if_present()

    repo_root = resolve_repo_root(Path(__file__).resolve().parent)
    default_uri = f"sqlite:///{repo_root / '.mlruns' / 'mlflow.db'}"

    if tracking_uri:
        uri = tracking_uri
        print(f"Using tracking URI: {uri}")
    elif os.getenv("MLFLOW_TRACKING_URI"):
        uri = os.getenv("MLFLOW_TRACKING_URI")
        print(f"Using tracking URI from environment: {uri}")
    else:
        uri = default_uri
        print(f"Using default tracking URI: {uri}")

    if experiment_name:
        exp_name = experiment_name
        print(f"Using MLFlow experiment name {exp_name}")
    elif os.getenv("MLFLOW_EXPERIMENT_NAME"):
        exp_name = os.getenv("MLFLOW_EXPERIMENT_NAME")
        print(f"Using MLFlow experiment name from environment: {exp_name}")
    else:
        exp_name = "diffmeshopt-2d"
        print(f"Using default MLFlow experiment name: {exp_name}")

    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(exp_name)
    return exp_name


def _is_primitive(v: Any) -> bool:
    return isinstance(v, (str, int, float, bool))


def _flatten(prefix: str, d: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flatten(key, v))
        else:
            out[key] = v
    return out


def _numeric_only(d: dict[str, Any], prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in d.items():
        key = f"{prefix}{k}" if prefix else k
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            out[key] = float(v)
    return out


def _find_run_by_logical_id(experiment_id: str, logical_run_id: str) -> str | None:
    runs = mlflow.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.diffmeshopt_run_id = '{logical_run_id}'",
        max_results=1,
        order_by=["attribute.start_time DESC"],
    )
    if runs.empty:
        return None
    return str(runs.iloc[0]["run_id"])


def _safe_log_params(params: dict[str, Any], *, only_if_new_run: bool) -> None:
    if not only_if_new_run:
        return
    clean = {k: str(v) for k, v in params.items() if v is not None and _is_primitive(v)}
    if clean:
        mlflow.log_params(clean)


def upsert_entry_to_mlflow(
    entry: ExperimentLogEntry,
    *,
    experiment_name: str = "diffmeshopt-2d",
    tracking_uri: str | None = None,
) -> str:
    """Upsert-like logging keyed by logical run_id tag.

    Behavior:
    - If run with tags.diffmeshopt_run_id exists -> reopen and update tags/metrics/artifacts.
    - Else -> create new run.
    - Params are logged only on first creation (MLflow params are immutable per key).
    """

    exp_name = _ensure_mlflow_configured(
        tracking_uri=tracking_uri, experiment_name=experiment_name
    )
    exp = mlflow.get_experiment_by_name(exp_name)
    assert exp is not None

    existing_run_id = _find_run_by_logical_id(exp.experiment_id, entry.run_id)
    is_new = existing_run_id is None

    if is_new:
        run_ctx = mlflow.start_run(run_name=entry.run_id)
    else:
        run_ctx = mlflow.start_run(run_id=existing_run_id)

    with run_ctx:
        run = mlflow.active_run()
        assert run is not None

        # Tags (mutable)
        mlflow.set_tags({
            "diffmeshopt_run_id": entry.run_id,
            "author": entry.author,
            "status": entry.status,
            "device": entry.device,
            "seeds": str(entry.seeds),
            "notebook_script": entry.notebook_script,
            "branch_commit": entry.branch_commit,
            "updated_at_utc": datetime.now(UTC).isoformat(),
        })

        if entry.goal:
            mlflow.set_tag("goal", entry.goal)
        if entry.dataset_path:
            mlflow.set_tag("dataset_path", entry.dataset_path)
        if entry.sample_key:
            mlflow.set_tag("sample_key", entry.sample_key)

        # Params (log only when run is first created)
        params = {
            **_flatten("config", entry.config),
            **_flatten("reg", entry.regularization_weights),
            "preprocessing": entry.preprocessing,
        }
        _safe_log_params(params, only_if_new_run=is_new)

        # Metrics (can be updated with new points)
        step = int(entry.runtime_notes.get("global_step", 0) or 0)
        metrics = {}
        metrics.update(_numeric_only(entry.quantitative_results))
        metrics.update(_numeric_only(entry.template_diagnostics, prefix="template."))
        metrics.update(_numeric_only(entry.runtime_notes, prefix="runtime."))
        if metrics:
            mlflow.log_metrics(metrics, step=step)

        # Artifacts: full snapshot (mutable via new versions)
        payload = {
            "run_id": entry.run_id,
            "entry": entry.__dict__,
        }
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            (p / "entry.json").write_text(json.dumps(payload, indent=2, default=str))
            (p / "entry.md").write_text(entry.to_markdown() + "\n")
            mlflow.log_artifacts(str(p), artifact_path="log_entry")

        return run.info.run_id


def upsert_trainer_run_to_mlflow(
    trainer: Any,
    run_id: str,
    *,
    experiment_name: str = "diffmeshopt-2d",
    tracking_uri: str | None = None,
    **kwargs: Any,
) -> str:
    entry = create_experiment_log_entry_from_trainer(trainer, run_id, **kwargs)
    return upsert_entry_to_mlflow(
        entry,
        experiment_name=experiment_name,
        tracking_uri=tracking_uri,
    )
