from __future__ import annotations

import torch


def compute_retrieval_metrics(
    score_matrix: torch.Tensor,
    query_ids: list[str],
    target_ids: list[str],
    *,
    higher_is_better: bool,
) -> dict[str, float]:
    if score_matrix.shape != (len(query_ids), len(target_ids)):
        raise ValueError(
            "score_matrix shape does not match ids: "
            f"{tuple(score_matrix.shape)} vs {len(query_ids)}x{len(target_ids)}"
        )
    sorted_indices = torch.argsort(score_matrix, dim=1, descending=higher_is_better).cpu()
    ranks: list[int] = []
    target_lookup: dict[str, list[int]] = {}
    for idx, video_id in enumerate(target_ids):
        target_lookup.setdefault(video_id, []).append(idx)

    for query_idx, video_id in enumerate(query_ids):
        positives = set(target_lookup.get(video_id, []))
        if not positives:
            continue
        row = sorted_indices[query_idx].tolist()
        ranks.append(min(row.index(pos) for pos in positives if pos in row))

    if not ranks:
        raise ValueError("No matching query/target ids were found for retrieval metrics.")

    rank_tensor = torch.tensor(ranks, dtype=torch.float32)
    return {
        "R@1": (rank_tensor < 1).float().mean().item() * 100.0,
        "R@5": (rank_tensor < 5).float().mean().item() * 100.0,
        "R@10": (rank_tensor < 10).float().mean().item() * 100.0,
        "MedR": rank_tensor.median().item() + 1.0,
        "MeanR": rank_tensor.mean().item() + 1.0,
        "n_queries": float(len(ranks)),
    }
