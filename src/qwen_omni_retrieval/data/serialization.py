from __future__ import annotations

from typing import Any

import torch


DTYPE_NAMES: dict[torch.dtype, str] = {
    torch.bool: "bool",
    torch.uint8: "uint8",
    torch.int8: "int8",
    torch.int16: "int16",
    torch.int32: "int32",
    torch.int64: "int64",
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float32: "float32",
    torch.float64: "float64",
}

NAME_DTYPES = {name: dtype for dtype, name in DTYPE_NAMES.items()}


def tensor_to_jsonable(tensor: torch.Tensor) -> dict[str, Any]:
    cpu_tensor = tensor.detach().cpu()
    return {
        "__tensor__": True,
        "dtype": DTYPE_NAMES.get(cpu_tensor.dtype, str(cpu_tensor.dtype).replace("torch.", "")),
        "shape": list(cpu_tensor.shape),
        "data": cpu_tensor.reshape(-1).tolist(),
    }


def tensor_from_jsonable(payload: dict[str, Any]) -> torch.Tensor:
    dtype = NAME_DTYPES[payload["dtype"]]
    return torch.tensor(payload["data"], dtype=dtype).reshape(payload["shape"])


def encode_jsonable(value: Any) -> Any:
    if torch.is_tensor(value):
        return tensor_to_jsonable(value)
    if isinstance(value, dict):
        return {key: encode_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [encode_jsonable(item) for item in value]
    return value


def decode_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        if value.get("__tensor__"):
            return tensor_from_jsonable(value)
        return {key: decode_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [decode_jsonable(item) for item in value]
    return value
