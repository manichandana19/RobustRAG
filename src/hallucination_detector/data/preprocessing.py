from typing import Any

from hallucination_detector.data.schema import RAGTruthRecord


def build_ragtruth_records(
    responses: list[dict[str, Any]],
    sources: list[dict[str, Any]],
) -> list[RAGTruthRecord]:
    """
    Join RAGTruth response records with their source records
    and convert them into the project's canonical schema.
    """

    source_by_id = {
        str(record["source_id"]): record
        for record in sources
    }

    records: list[RAGTruthRecord] = []

    for response in responses:
        source_id = str(response["source_id"])

        if source_id not in source_by_id:
            raise ValueError(
                f"Missing source_id={source_id} for "
                f"response id={response.get('id')}"
            )

        source = source_by_id[source_id]

        labels = response.get("labels", [])

        if not isinstance(labels, list):
            raise ValueError(
                f"Expected labels to be a list for "
                f"response id={response.get('id')}"
            )

        question = source.get("prompt", "")
        context = source.get("source_info", "")
        answer = response.get("response", "")

        if not question or not context or not answer:
            raise ValueError(
                f"Missing required text field for "
                f"response id={response.get('id')}"
            )

        records.append(
            RAGTruthRecord(
                id=str(response["id"]),
                source_id=source_id,
                question=question,
                context=context,
                answer=answer,
                hallucinated=len(labels) > 0,
                model=response.get("model"),
                temperature=response.get("temperature"),
                split=response["split"],
                quality=response.get("quality"),
                annotations=labels,
            )
        )

    return records