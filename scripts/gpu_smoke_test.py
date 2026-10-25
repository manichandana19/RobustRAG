"""Run one cached RobustRAG transformer inference on CUDA and compare with CPU.

This is a manual smoke test, not a benchmark. It does not train models or
modify checkpoints, thresholds, or production inference code.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from huggingface_hub import scan_cache_dir

from hallucination_detector.inference.model_registry import MODEL_CATALOG, ModelRegistry


CASE_PATH = ROOT / "reports" / "manual_evaluation" / "robustrag_manual_set_v1.json"
CASE = next(item for item in json.loads(CASE_PATH.read_text(encoding="utf-8"))["cases"]
            if item["case_id"] == "v1_qa_contradiction_location")
QUESTION, CONTEXT, ANSWER = (CASE["question"], CASE["context"], CASE["answer"])


def cuda_memory(device: torch.device) -> dict[str, float]:
    index = device.index or 0
    return {
        "allocated_mib": round(torch.cuda.memory_allocated(index) / 2**20, 2),
        "reserved_mib": round(torch.cuda.memory_reserved(index) / 2**20, 2),
        "peak_allocated_mib": round(torch.cuda.max_memory_allocated(index) / 2**20, 2),
        "peak_reserved_mib": round(torch.cuda.max_memory_reserved(index) / 2**20, 2),
    }


def is_cuda_oom(error: Exception) -> bool:
    return isinstance(error, torch.cuda.OutOfMemoryError) or (
        isinstance(error, RuntimeError) and "out of memory" in str(error).lower()
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", choices=("deberta", "longformer-head", "longformer-qca"),
        default="deberta", help="One registered transformer model (default: deberta).",
    )
    parser.add_argument("--device", default="cuda:0", help="CUDA device to smoke test.")
    parser.add_argument(
        "--skip-cpu", action="store_true", help="Skip same-input CPU comparison."
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print(json.dumps({"status": "blocked", "reason": "CUDA is unavailable."}))
        return 2

    device = torch.device(args.device)
    if device.type != "cuda":
        parser.error("--device must select a CUDA device")
    torch.cuda.set_device(device)
    revisions = {
        repo.repo_id: [revision.commit_hash for revision in repo.revisions]
        for repo in scan_cache_dir().repos
    }
    metadata = MODEL_CATALOG[args.model]
    report: dict[str, object] = {
        "model_key": args.model,
        "model_id": metadata.repo_id,
        "cached_revisions": revisions.get(metadata.repo_id, []),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(device),
        "device": str(device),
        "question": QUESTION,
        "context": CONTEXT,
        "answer": ANSWER,
        "cpu_comparison": "skipped" if args.skip_cpu else "pending",
    }

    registry: ModelRegistry | None = ModelRegistry()
    model_obj = None
    model = None
    result = None
    torch.cuda.reset_peak_memory_stats(device)
    try:
        model_obj = registry.load_model(args.model, device=device)
        model = model_obj["model"]
        parameter_device = str(next(model.parameters()).device)
        report["parameter_device"] = parameter_device
        if torch.device(parameter_device) != device:
            raise RuntimeError(
                f"Requested {device}, but model parameters are on {parameter_device}."
            )
        report["memory_after_load"] = cuda_memory(device)

        result = registry.predict(
            model_key=args.model,
            question=QUESTION,
            context=CONTEXT,
            answer=ANSWER,
            device=device,
        )
        torch.cuda.synchronize(device)
        report["cuda_result"] = {
            "prediction": result.prediction,
            "supported_probability": result.supported_probability,
            "hallucination_probability": result.hallucination_probability,
            "details": result.details,
        }
        report["memory_after_inference"] = cuda_memory(device)
        report["status"] = "ok"
    except Exception as error:
        if not is_cuda_oom(error):
            raise
        report.update(status="cuda_oom", error_type=type(error).__name__, error=str(error))
        result = None
    finally:
        del model_obj
        del model
        del registry
        gc.collect()
        torch.cuda.empty_cache()
        report["memory_after_cleanup"] = {
            "allocated_mib": round(torch.cuda.memory_allocated(device) / 2**20, 2),
            "reserved_mib": round(torch.cuda.memory_reserved(device) / 2**20, 2),
        }

    if result is not None and not args.skip_cpu:
        cpu_registry = ModelRegistry()
        cpu_model_obj = cpu_registry.load_model(args.model, device="cpu")
        report["cpu_parameter_device"] = str(next(cpu_model_obj["model"].parameters()).device)
        cpu_result = cpu_registry.predict(
            model_key=args.model,
            question=QUESTION,
            context=CONTEXT,
            answer=ANSWER,
            device="cpu",
        )
        report["cpu_result"] = {
            "prediction": cpu_result.prediction,
            "supported_probability": cpu_result.supported_probability,
            "hallucination_probability": cpu_result.hallucination_probability,
        }
        report["absolute_score_differences"] = {
            "supported": abs(result.supported_probability - cpu_result.supported_probability),
            "hallucination": abs(
                result.hallucination_probability - cpu_result.hallucination_probability
            ),
        }
        report["cpu_comparison"] = "completed"

    print(json.dumps(report, indent=2, default=str))
    return 2 if report["status"] == "cuda_oom" else 0


if __name__ == "__main__":
    raise SystemExit(main())
