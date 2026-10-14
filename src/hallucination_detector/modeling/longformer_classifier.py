from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence, Literal

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (
    LongformerForSequenceClassification,
    AutoTokenizer,
)

from hallucination_detector.data.schema import RAGTruthRecord
from hallucination_detector.data.torch_dataset import (
    RAGTruthTransformerDataset,
    LongformerDataCollator,
)


@dataclass
class LongformerClassifier:
    """Binary Longformer classifier for supported vs. hallucinated answers.

    Label convention:
        0 = supported
        1 = hallucinated
    """

    model_name: str = "allenai/longformer-base-4096"
    max_length: int = 1024
    input_mode: str = "ca"
    truncation_strategy: Literal["head", "tail"] = "head"
    global_attention_mode: Literal["cls", "cls_and_answer"] = "cls"
    num_workers: int = 0
    tokenizer: object = field(init=False, repr=False)
    model: object = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.max_length <= 0:
            raise ValueError("max_length must be a positive integer.")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative.")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
        )

        self.model = LongformerForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=2,
            torch_dtype=torch.float32,
            id2label={0: "SUPPORTED", 1: "HALLUCINATED"},
            label2id={"SUPPORTED": 0, "HALLUCINATED": 1},
        )
        self._assert_fp32()

    def _assert_fp32(self) -> None:
        non_fp32 = [
            (name, str(param.dtype))
            for name, param in self.model.named_parameters()
            if param.is_floating_point() and param.dtype != torch.float32
        ]
        if non_fp32:
            raise RuntimeError(
                "Model parameters are not all FP32: "
                f"{non_fp32[:10]}"
            )

    def enable_gradient_checkpointing(self) -> None:
        """Enable activation checkpointing to reduce training memory use."""
        self.model.gradient_checkpointing_enable()

    def build_dataset(
        self,
        records: Sequence[RAGTruthRecord],
    ) -> RAGTruthTransformerDataset:
        return RAGTruthTransformerDataset(
            records=records,
            tokenizer=self.tokenizer,
            input_mode=self.input_mode,
            max_length=self.max_length,
            truncation_strategy=self.truncation_strategy,
            global_attention_mode=self.global_attention_mode,
        )

    def save_pretrained(self, output_dir: str | Path) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        self._assert_fp32()
        self.model.config.id2label = {
            0: "SUPPORTED",
            1: "HALLUCINATED",
        }
        self.model.config.label2id = {
            "SUPPORTED": 0,
            "HALLUCINATED": 1,
        }
        self.model.save_pretrained(str(output_path))
        self.tokenizer.save_pretrained(str(output_path))

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str | Path,
        max_length: int = 1024,
        input_mode: str = "ca",
        truncation_strategy: Literal["head", "tail"] = "head",
        global_attention_mode: Literal["cls", "cls_and_answer"] = "cls",
        num_workers: int = 0,
    ) -> "LongformerClassifier":
        """Load a checkpoint saved by this class."""
        model_path = str(model_dir)
        classifier = cls.__new__(cls)
        classifier.model_name = model_path
        classifier.max_length = max_length
        classifier.input_mode = input_mode
        classifier.truncation_strategy = truncation_strategy
        classifier.global_attention_mode = global_attention_mode
        classifier.num_workers = num_workers

        if max_length <= 0:
            raise ValueError("max_length must be a positive integer.")
        if num_workers < 0:
            raise ValueError("num_workers cannot be negative.")

        classifier.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
        )

        classifier.model = LongformerForSequenceClassification.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
        )
        classifier._assert_fp32()
        return classifier

    def predict_proba(
        self,
        records: Sequence[RAGTruthRecord],
        batch_size: int = 4,
        device: str | torch.device | None = None,
        bf16: bool = False,
    ) -> np.ndarray:
        """Return probabilities shaped (n_records, 2).

        Column 0 is P(supported); column 1 is P(hallucinated).
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")
        if not records:
            return np.empty((0, 2), dtype=np.float32)

        if device is None:
            device_obj = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            device_obj = torch.device(device)

        use_bf16 = (
            bf16
            and device_obj.type == "cuda"
            and torch.cuda.is_bf16_supported()
        )

        self.model.to(device_obj)
        self.model.eval()

        dataset = self.build_dataset(records)
        collator = LongformerDataCollator(
            tokenizer=self.tokenizer,
            pad_to_multiple_of=512,
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=self.num_workers,
            pin_memory=device_obj.type == "cuda",
        )

        probability_batches: list[np.ndarray] = []
        with torch.inference_mode():
            for batch in loader:
                batch.pop("labels", None)
                batch = {
                    key: value.to(device_obj, non_blocking=device_obj.type == "cuda")
                    for key, value in batch.items()
                }

                if use_bf16:
                    with torch.autocast(
                        device_type="cuda",
                        dtype=torch.bfloat16,
                    ):
                        outputs = self.model(**batch)
                else:
                    outputs = self.model(**batch)

                probs = torch.softmax(outputs.logits.float(), dim=-1)
                probability_batches.append(probs.cpu().numpy())

        return np.concatenate(probability_batches, axis=0)

    def predict(
        self,
        records: Sequence[RAGTruthRecord],
        batch_size: int = 4,
        device: str | torch.device | None = None,
        bf16: bool = False,
    ) -> np.ndarray:
        probabilities = self.predict_proba(
            records=records,
            batch_size=batch_size,
            device=device,
            bf16=bf16,
        )
        return np.argmax(probabilities, axis=1)
