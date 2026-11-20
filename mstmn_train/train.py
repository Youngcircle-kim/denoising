"""Auditable unofficial MSTMN retraining with atomic resumable checkpoints.

Run ``python -m mstmn_train.train --help``. PyTorch is imported only when training
starts. The JSON recipe records paper settings separately from declared choices.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time


ARTIFACT_TYPE = "mstmn-unofficial-retrain"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _positive_int(value: int, name: str, *, zero: bool = False) -> int:
    if type(value) is not int or value < (0 if zero else 1):
        raise ValueError(f"{name} must be an integer >= {0 if zero else 1}")
    return value


def validate_config(config: dict) -> None:
    if config.get("schema_version") != 1 or config.get("artifact_type") != ARTIFACT_TYPE:
        raise ValueError("Expected schema1 mstmn-unofficial-retrain config")
    training, optimizer, reproduction = (config[name] for name in ("training", "optimizer", "reproducibility"))
    for name in ("total_steps", "batch_size", "checkpoint_interval", "log_interval"):
        _positive_int(training[name], name)
    _positive_int(training["num_workers"], "num_workers", zero=True)
    _positive_int(reproduction["seed"], "seed", zero=True)
    if training["loss"] != "mean_l1_all_frames" or training["bptt"] != "full_sequence_without_detach":
        raise ValueError("Only mean L1 with full sequence BPTT is supported")
    if optimizer["name"] != "Adam" or optimizer["weight_decay"] != 0:
        raise ValueError("Recipe requires Adam with weight_decay0")
    if len(optimizer["betas"]) != 2 or not all(0 <= beta < 1 for beta in optimizer["betas"]):
        raise ValueError("Adam betas must be in [0,1)")
    if optimizer["eps"] <= 0 or optimizer["learning_rate"] <= 0:
        raise ValueError("Adam epsilon and learning rate must be positive")
    previous = 0
    for milestone, rate in optimizer["schedule"]:
        _positive_int(milestone, "milestone")
        if milestone <= previous or not math.isfinite(rate) or rate <= 0:
            raise ValueError("Learning-rate milestones must increase and rates must be positive")
        previous = milestone
    if reproduction["precision"] != "float32" or reproduction["amp"] or reproduction["tf32"]:
        raise ValueError("This engine requires FP32 without AMP/TF32")
    # CUDA grid_sample backward is not deterministic; do not quietly promise it.
    if reproduction["deterministic_algorithms"]:
        raise ValueError("Strict deterministic CUDA algorithms are unsupported by the selected warp path")
    _positive_int(config["model"]["memory_frames"], "memory_frames")


def learning_rate_for_step(step: int, optimizer_config: dict) -> float:
    """One-based update number; the milestone update uses the new rate."""
    _positive_int(step, "step")
    rate = float(optimizer_config["learning_rate"])
    for milestone, next_rate in optimizer_config["schedule"]:
        if step >= milestone:
            rate = float(next_rate)
    return rate


def sample_window(completed_steps: int, maximum_steps: int, batch_size: int) -> tuple[int, int]:
    """Consumed global sample count and the remaining dataset length."""
    _positive_int(completed_steps, "completed_steps", zero=True)
    _positive_int(maximum_steps, "maximum_steps")
    _positive_int(batch_size, "batch_size")
    if completed_steps > maximum_steps:
        raise ValueError("Resume checkpoint is beyond the requested stopping step")
    return completed_steps * batch_size, (maximum_steps - completed_steps) * batch_size


def train_effective_batch(model, optimizer, batch: dict, *, microbatch: int, memory_frames: int, device) -> float:
    """Full sequence BPTT; accumulate microbatches into one effective update.

    No previous restoration is detached, and no gradient clipping is applied.
    Frozen flow inference is owned by the model. Return the effective-batch mean
    over samples, frames, channels and spatial positions.
    """
    import torch
    from torch.nn import functional as F

    _positive_int(microbatch, "microbatch")
    inputs, targets = batch["input"], batch["target"]
    if inputs.ndim != 5 or inputs.shape != targets.shape or inputs.shape[2] != 4:
        raise ValueError("Training input and target must match B,T,4,H,W")
    batch_size, sequence_length = inputs.shape[:2]
    if batch_size < 1 or sequence_length < 1:
        raise ValueError("Empty training batch")
    optimizer.zero_grad(set_to_none=True)
    loss_value = 0.0
    for start in range(0, batch_size, microbatch):
        end = min(start + microbatch, batch_size)
        noisy = inputs[start:end].to(device=device, dtype=torch.float32, non_blocking=True)
        clean = targets[start:end].to(device=device, dtype=torch.float32, non_blocking=True)
        shot = batch["shot"][start:end].to(device=device, dtype=torch.float32, non_blocking=True)
        read = batch["read"][start:end].to(device=device, dtype=torch.float32, non_blocking=True)
        previous = noisy[:, 0]
        history = previous[:, None].repeat(1, memory_frames, 1, 1, 1)
        sequence_loss = noisy.new_zeros(())
        for frame in range(sequence_length):
            current = noisy[:, frame]
            prediction = model(current, previous, history, shot, read)
            if prediction.shape != clean[:, frame].shape:
                raise ValueError("Model prediction shape does not match packed RAW target")
            sequence_loss = sequence_loss + F.l1_loss(prediction, clean[:, frame], reduction="mean")
            previous = prediction
            history = torch.cat((history[:, 1:], current[:, None]), dim=1)
        weighted_loss = sequence_loss / sequence_length * ((end - start) / batch_size)
        if not bool(torch.isfinite(weighted_loss)):
            raise FloatingPointError("Nonfinite training loss; optimizer update was not performed")
        weighted_loss.backward()
        loss_value += float(weighted_loss.detach())
    finite_gradients = [torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None]
    if not finite_gradients or not bool(torch.stack(finite_gradients).all()):
        raise FloatingPointError("Missing/nonfinite gradients; optimizer update was not performed")
    optimizer.step()
    return loss_value


def source_hashes() -> dict[str, str]:
    folder = Path(__file__).resolve().parent
    hashes = {name: sha256_file(folder / name) for name in ("model.py", "data.py", "flow.py", "train.py")}
    hashes["denoising_eval/data.py"] = sha256_file(folder.parent / "denoising_eval/data.py")
    return hashes


def _save_checkpoint(path: Path, value: dict, torch) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=".checkpoint.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def validate_resume(checkpoint: dict, config: dict, flow_sha256: str, hashes: dict) -> int:
    if checkpoint.get("schema_version") != 1 or checkpoint.get("artifact_type") != ARTIFACT_TYPE:
        raise ValueError("Unsupported/unofficial-training checkpoint type")
    if checkpoint["config"] != config:
        raise ValueError("Resume config does not exactly match the saved recipe")
    if checkpoint["flow"]["sha256"] != flow_sha256:
        raise ValueError("Resume flow checkpoint SHA-256 mismatch")
    if checkpoint["source_hashes"] != hashes:
        raise ValueError("Resume implementation source hashes changed")
    completed = _positive_int(checkpoint["completed_steps"], "completed_steps", zero=True)
    if completed > config["training"]["total_steps"]:
        raise ValueError("Checkpoint exceeds the recipe's total_steps")
    return completed


def run_training(args: argparse.Namespace) -> dict:
    config_path = Path(args.config).expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    maximum_steps = args.max_steps if args.max_steps is not None else config["training"]["total_steps"]
    _positive_int(maximum_steps, "max_steps")
    if maximum_steps > config["training"]["total_steps"]:
        raise ValueError("max_steps may stop early but cannot exceed the saved recipe")
    batch_size = config["training"]["batch_size"]
    microbatch = batch_size if args.microbatch is None else _positive_int(args.microbatch, "microbatch")
    if microbatch > batch_size:
        raise ValueError("microbatch cannot exceed effective batch_size")
    data_root = Path(args.data_root).expanduser().resolve()
    flow_path = Path(args.flow_checkpoint).expanduser().resolve()
    if not flow_path.is_file():
        raise FileNotFoundError(flow_path)
    output = Path(args.output).expanduser().resolve()
    resume = Path(args.resume).expanduser().resolve() if args.resume else None
    if resume is None:
        output.mkdir(parents=True, exist_ok=False)
    elif not resume.is_file() or resume.parent != output or not output.is_dir():
        raise ValueError("Resume must be an existing checkpoint inside the selected output directory")
    lock = output / ".train.lock"
    with lock.open("x", encoding="utf-8") as handle:
        json.dump({"pid": os.getpid(), "started_at_utc": utc_now()}, handle)
    completed_steps, checkpoint_steps = 0, None
    started = time.perf_counter()
    status = {
        "artifact_type": ARTIFACT_TYPE, "state": "starting", "pid": os.getpid(),
        "started_at_utc": utc_now(), "updated_at_utc": utc_now(),
        "completed_steps": 0, "run_max_steps": maximum_steps,
        "recipe_total_steps": config["training"]["total_steps"],
        "effective_batch_size": batch_size, "microbatch_size": microbatch,
        "data_root": str(data_root), "checkpoint": str(output / "last.pth"),
    }
    try:
        atomic_json(output / "status.json", status)
        import torch
        from .data import CRVDTrainingDataset
        from .flow import RawSPyNet
        from .model import MSTMN

        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; no CPU fallback was started")
        reproduction = config["reproducibility"]
        seed = reproduction["seed"]
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = reproduction["cudnn_benchmark"]
        torch.backends.cudnn.deterministic = reproduction["cudnn_deterministic"]
        torch.use_deterministic_algorithms(False)
        torch.set_float32_matmul_precision("highest")
        flow = {"path": str(flow_path), "sha256": sha256_file(flow_path)}
        hashes = source_hashes()
        model = MSTMN(flow_model=RawSPyNet(flow_path), **config["model"]).to(device=device, dtype=torch.float32)
        model.train()
        optimizer_config = config["optimizer"]
        optimizer = torch.optim.Adam(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=optimizer_config["learning_rate"], betas=tuple(optimizer_config["betas"]),
            eps=optimizer_config["eps"], weight_decay=optimizer_config["weight_decay"],
        )
        if resume:
            saved = torch.load(resume, map_location="cpu", weights_only=True)
            completed_steps = validate_resume(saved, config, flow["sha256"], hashes)
            if saved["data_root"] != str(data_root):
                raise ValueError("Resume data_root differs from the saved run")
            model.load_state_dict(saved["model_state_dict"], strict=True)
            optimizer.load_state_dict(saved["optimizer_state_dict"])
            torch.set_rng_state(saved["rng_cpu"])
            if device.type == "cuda":
                if len(saved["rng_cuda"]) != torch.cuda.device_count():
                    raise ValueError("Resume CUDA device count differs from the saved run")
                torch.cuda.set_rng_state_all(saved["rng_cuda"])
            checkpoint_steps = completed_steps
        start_index, samples = sample_window(completed_steps, maximum_steps, batch_size)
        status.update({"state": "running", "completed_steps": completed_steps,
                       "flow": flow, "source_hashes": hashes, "torch_version": str(torch.__version__),
                       "device": str(device), "cuda_runtime": torch.version.cuda,
                       "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                       "last_checkpoint_steps": checkpoint_steps})
        atomic_json(output / "recipe.json", config)
        atomic_json(output / "status.json", status)

        def checkpoint() -> None:
            nonlocal checkpoint_steps
            _save_checkpoint(output / "last.pth", {
                "schema_version": 1, "artifact_type": ARTIFACT_TYPE,
                "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                "completed_steps": completed_steps, "config": config, "flow": flow,
                "model_implementation": model.model_config,
                "rng_cpu": torch.get_rng_state(),
                "rng_cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                "seed": seed, "source_hashes": hashes, "data_root": str(data_root),
                "run_max_steps": maximum_steps, "microbatch_size": microbatch, "saved_at_utc": utc_now(),
            }, torch)
            checkpoint_steps = completed_steps

        stopped = False
        if samples:
            dataset = CRVDTrainingDataset(data_root, samples=samples, seed=seed, start_index=start_index, **config["dataset"])
            workers = config["training"]["num_workers"]
            # A separate generator prevents DataLoader worker seeding from
            # consuming/restoring the model's RNG stream on a resumed run.
            loader_generator = torch.Generator().manual_seed(seed)
            loader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
                drop_last=True, pin_memory=device.type == "cuda", persistent_workers=workers > 0,
                generator=loader_generator,
            )
            batches = iter(loader)
            with (output / "train.jsonl").open("a", encoding="utf-8", buffering=1) as log:
                while completed_steps < maximum_steps:
                    if (output / "STOP").exists():
                        stopped = True
                        break
                    step_started = time.perf_counter()
                    if device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats(device)
                    batch = next(batches)
                    step = completed_steps + 1
                    rate = learning_rate_for_step(step, optimizer_config)
                    for group in optimizer.param_groups:
                        group["lr"] = rate
                    loss = train_effective_batch(model, optimizer, batch, microbatch=microbatch,
                                                 memory_frames=config["model"]["memory_frames"], device=device)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    completed_steps = step
                    if step == 1 or step % config["training"]["checkpoint_interval"] == 0:
                        checkpoint()
                    observed = {
                        "at_utc": utc_now(), "step": step, "loss": loss, "learning_rate": rate,
                        "step_seconds": time.perf_counter() - step_started,
                        "run_seconds": time.perf_counter() - started,
                        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                        "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
                    }
                    if step == 1 or step % config["training"]["log_interval"] == 0 or step == maximum_steps:
                        log.write(json.dumps(observed, allow_nan=False) + "\n")
                        log.flush()
                        print(json.dumps(observed, allow_nan=False), flush=True)
                    status.update({"updated_at_utc": utc_now(), "completed_steps": completed_steps,
                                   "last_checkpoint_steps": checkpoint_steps, "last_observation": observed})
                    atomic_json(output / "status.json", status)
        checkpoint()
        status.update({"state": "stopped" if stopped else "completed", "updated_at_utc": utc_now(),
                       "completed_steps": completed_steps, "last_checkpoint_steps": checkpoint_steps,
                       "run_seconds": time.perf_counter() - started})
        atomic_json(output / "status.json", status)
        return status
    except BaseException as error:
        # Never publish potentially half-updated optimizer/model state. The last
        # atomic checkpoint remains intact; the status exposes its exact step.
        status.update({"state": "failed", "updated_at_utc": utc_now(), "completed_steps": completed_steps,
                       "last_checkpoint_steps": checkpoint_steps, "error_type": type(error).__name__,
                       "error": str(error), "run_seconds": time.perf_counter() - started})
        atomic_json(output / "status.json", status)
        raise
    finally:
        lock.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/mstmn_retrain.json"))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--flow-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, help="Stop early while retaining the original full recipe")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--microbatch", type=int, help="Accumulate this many samples at a time into the configured effective batch")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    args = parser.parse_args(argv)
    run_training(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
