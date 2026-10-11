# from __future__ import annotations

# from dataclasses import dataclass
# from typing import Sequence

# import numpy as np
# import torch
# from transformers import (
#     AutoModelForSequenceClassification,
#     AutoTokenizer,
# )

# from hallucination_detector.data.schema import RAGTruthRecord
# from hallucination_detector.data.torch_dataset import RAGTruthTransformerDataset


# @dataclass
# class DeBERTaClassifier:
#     """
#     DeBERTa-v3-base sequence classifier.

#     Architecture:

#         DeBERTa-v3-base encoder
#                 |
#         transformer classification head
#                 |
#              2 logits
#                 |
#             softmax
#                 |
#         P(supported), P(hallucinated)

#     Model parameters are explicitly loaded and saved in FP32.
#     Training may use BF16 autocast for memory-efficient computation.
#     """

#     model_name: str = "microsoft/deberta-v3-base"
#     max_length: int = 512
#     input_mode: str = "ca"

#     def __post_init__(self) -> None:
#         self.tokenizer = AutoTokenizer.from_pretrained(
#             self.model_name,
#             use_fast=True,
#         )

#         self.model = AutoModelForSequenceClassification.from_pretrained(
#             self.model_name,
#             num_labels=2,
#             dtype=torch.float32,
#         )

#         self._assert_fp32()

#     def _assert_fp32(self) -> None:
#         """Fail loudly if model parameters are not stored in FP32."""
#         non_fp32 = [
#             (name, param.dtype)
#             for name, param in self.model.named_parameters()
#             if param.is_floating_point()
#             and param.dtype != torch.float32
#         ]

#         if non_fp32:
#             raise RuntimeError(
#                 "DeBERTa model contains non-FP32 parameters: "
#                 f"{non_fp32[:10]}"
#             )

#     def enable_gradient_checkpointing(self) -> None:
#         """Enable activation checkpointing for memory-efficient training."""
#         self.model.gradient_checkpointing_enable()

#     def build_dataset(
#         self,
#         records: Sequence[RAGTruthRecord],
#     ) -> RAGTruthTransformerDataset:
#         return RAGTruthTransformerDataset(
#             records=records,
#             tokenizer=self.tokenizer,
#             input_mode=self.input_mode,
#             max_length=self.max_length,
#         )

#     def save_pretrained(self, output_dir: str) -> None:
#         self._assert_fp32()
#         self.model.save_pretrained(output_dir)
#         self.tokenizer.save_pretrained(output_dir)

#     @classmethod
#     def from_pretrained(
#         cls,
#         model_dir: str,
#         max_length: int = 512,
#         input_mode: str = "ca",
#     ) -> "DeBERTaClassifier":
#         classifier = cls.__new__(cls)

#         classifier.model_name = model_dir
#         classifier.max_length = max_length
#         classifier.input_mode = input_mode

#         classifier.tokenizer = AutoTokenizer.from_pretrained(
#             model_dir,
#             use_fast=True,
#         )

#         classifier.model = AutoModelForSequenceClassification.from_pretrained(
#             model_dir,
#             dtype=torch.float32,
#         )

#         classifier._assert_fp32()

#         return classifier

#     def predict_proba(
#         self,
#         records: Sequence[RAGTruthRecord],
#         batch_size: int = 8,
#         device: str | None = None,
#         bf16: bool = False,
#     ) -> np.ndarray:
#         """
#         Return probabilities with shape (n_records, 2).

#         Column 0 = P(supported)
#         Column 1 = P(hallucinated)
#         """

#         if device is None:
#             device = "cuda" if torch.cuda.is_available() else "cpu"

#         self.model.to(device)
#         self.model.eval()

#         dataset = self.build_dataset(records)

#         loader = torch.utils.data.DataLoader(
#             dataset,
#             batch_size=batch_size,
#             shuffle=False,
#         )

#         use_bf16 = (
#             bf16
#             and device.startswith("cuda")
#             and torch.cuda.is_bf16_supported()
#         )

#         probabilities: list[np.ndarray] = []

#         with torch.no_grad():
#             for batch in loader:
#                 batch.pop("labels")

#                 batch = {
#                     key: value.to(device)
#                     for key, value in batch.items()
#                 }

#                 if use_bf16:
#                     with torch.autocast(
#                         device_type="cuda",
#                         dtype=torch.bfloat16,
#                     ):
#                         outputs = self.model(**batch)
#                 else:
#                     outputs = self.model(**batch)

#                 probs = torch.softmax(outputs.logits, dim=-1)

#                 probabilities.append(
#                     probs.detach().cpu().to(torch.float32).numpy()

#                 )

#         if not probabilities:
#             return np.empty((0, 2), dtype=np.float32)

#         return np.concatenate(probabilities, axis=0)

#     def predict(
#         self,
#         records: Sequence[RAGTruthRecord],
#         batch_size: int = 8,
#         device: str | None = None,
#         bf16: bool = False,
#     ) -> np.ndarray:
#         probabilities = self.predict_proba(
#             records=records,
#             batch_size=batch_size,
#             device=device,
#             bf16=bf16,
#         )

#         return np.argmax(probabilities, axis=1)from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)

from hallucination_detector.data.schema import RAGTruthRecord
from hallucination_detector.data.torch_dataset import RAGTruthTransformerDataset


SPECIAL_TOKENS = ["[CONTEXT]", "[ANSWER]"]


@dataclass
class DeBERTaClassifier:
    """Binary DeBERTa classifier for supported vs. hallucinated answers.

    Expected label convention:
        0 = supported
        1 = hallucinated

    IMPORTANT: RAGTruthTransformerDataset must use this same label convention
    and must include [CONTEXT] / [ANSWER] markers in its encoded input when
    the selected input_mode requires them.
    """

    model_name: str = "microsoft/deberta-v3-base"
    max_length: int = 512
    input_mode: str = "ca"
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
            use_fast=True,
        )
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": SPECIAL_TOKENS}
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=2,
            torch_dtype=torch.float32,
            id2label={0: "SUPPORTED", 1: "HALLUCINATED"},
            label2id={"SUPPORTED": 0, "HALLUCINATED": 1},
        )
        self.model.resize_token_embeddings(len(self.tokenizer))
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
        # Disable KV caching during checkpointed training.
        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = False

    def build_dataset(
        self,
        records: Sequence[RAGTruthRecord],
    ) -> RAGTruthTransformerDataset:
        return RAGTruthTransformerDataset(
            records=records,
            tokenizer=self.tokenizer,
            input_mode=self.input_mode,
            max_length=self.max_length,
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
        max_length: int = 512,
        input_mode: str = "ca",
        num_workers: int = 0,
    ) -> "DeBERTaClassifier":
        """Load a checkpoint saved by this class."""
        model_path = str(model_dir)
        classifier = cls.__new__(cls)
        classifier.model_name = model_path
        classifier.max_length = max_length
        classifier.input_mode = input_mode
        classifier.num_workers = num_workers

        if max_length <= 0:
            raise ValueError("max_length must be a positive integer.")
        if num_workers < 0:
            raise ValueError("num_workers cannot be negative.")

        classifier.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=True,
        )
        missing_tokens = [
            token
            for token in SPECIAL_TOKENS
            if token not in classifier.tokenizer.get_vocab()
        ]
        if missing_tokens:
            raise ValueError(
                "Checkpoint tokenizer is missing required special tokens "
                f"{missing_tokens}. Load a checkpoint saved with this class "
                "or retrain/save the tokenizer and model together."
            )

        classifier.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
        )
        embedding_count = classifier.model.get_input_embeddings().num_embeddings
        if embedding_count != len(classifier.tokenizer):
            raise ValueError(
                "Tokenizer/model vocabulary mismatch: tokenizer has "
                f"{len(classifier.tokenizer)} tokens but model embeddings have "
                f"{embedding_count} rows. Save/load the model and tokenizer "
                "from the same checkpoint."
            )

        classifier._assert_fp32()
        return classifier

    def predict_proba(
        self,
        records: Sequence[RAGTruthRecord],
        batch_size: int = 8,
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
        collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
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
        batch_size: int = 8,
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
