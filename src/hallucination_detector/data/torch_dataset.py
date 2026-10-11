# from __future__ import annotations

# from typing import Any, Sequence

# import torch
# from torch.utils.data import Dataset
# from transformers import PreTrainedTokenizerBase

# from hallucination_detector.data.schema import RAGTruthRecord
# from hallucination_detector.modeling.tfidf_logistic import build_input_text


# class RAGTruthTransformerDataset(Dataset):
#     """
#     PyTorch Dataset for DeBERTa-based hallucination classification.

#     Input representation is intentionally identical to the classical
#     C+A experiment:

#         [CONTEXT] <context> [ANSWER] <answer>

#     Labels:
#         0 = supported
#         1 = hallucinated
#     """

#     def __init__(
#         self,
#         records: Sequence[RAGTruthRecord],
#         tokenizer: PreTrainedTokenizerBase,
#         input_mode: str = "ca",
#         max_length: int = 512,
#     ) -> None:
#         if input_mode != "ca":
#             raise ValueError(
#                 "The DeBERTa baseline currently supports only input_mode='ca'."
#             )

#         if max_length <= 0:
#             raise ValueError("max_length must be positive.")

#         self.records = list(records)
#         self.tokenizer = tokenizer
#         self.input_mode = input_mode
#         self.max_length = max_length

#     def __len__(self) -> int:
#         return len(self.records)

#     def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
#         record = self.records[index]

#         text = build_input_text(record, self.input_mode)

#         # encoded = self.tokenizer(
#         #     text,
#         #     max_length=self.max_length,
#         #     truncation=True,
#         #     padding="max_length",
#         #     return_attention_mask=True,
#         #     return_tensors="pt",
#         # )
#         encoded = self.tokenizer(
#             text,
#             max_length=self.max_length,
#             truncation=True,
#             padding=False,
#             return_attention_mask=True,
#         )
#         item = {
#             key: value.squeeze(0)
#             for key, value in encoded.items()
#         }

#         item["labels"] = torch.tensor(
#             int(record.hallucinated),
#             dtype=torch.long,
#         )

#         return item
from __future__ import annotations
from typing import Any, Sequence
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase
from hallucination_detector.data.schema import RAGTruthRecord
from hallucination_detector.modeling.tfidf_logistic import build_input_text

class RAGTruthTransformerDataset(Dataset):
    def __init__(
        self,
        records: Sequence[RAGTruthRecord],
        tokenizer: PreTrainedTokenizerBase,
        input_mode: str = "ca",
        max_length: int = 512,
    ) -> None:
        if input_mode != "ca":
            raise ValueError("The DeBERTa baseline currently supports only input_mode='ca'.")
        if max_length <= 0:
            raise ValueError("max_length must be positive.")

        self.records = list(records)
        self.tokenizer = tokenizer
        self.input_mode = input_mode
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        text = build_input_text(record, self.input_mode)

        # Optimization: Remove padding="max_length" here. Pad dynamically in DataCollator.
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding=False,  # Changed from "max_length"
            return_attention_mask=True,
        )

        item = {key: torch.tensor(value) for key, value in encoded.items()}
        item["labels"] = torch.tensor(int(record.hallucinated), dtype=torch.long)
        return item
