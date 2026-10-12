"""Compare QSA Triton padding bounds with an unmodified baseline source file."""

import argparse
import importlib
import importlib.util
import json
import statistics
import time
from pathlib import Path

import torch
import torch_npu


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--only-wide-pool", action="store_true")
    args = parser.parse_args()
    name = "sgl_kernel_npu.qwen3_8_flash_next.qsa_sparse_attention"
    after = importlib.import_module(name)
    spec = importlib.util.spec_from_file_location(name + "_baseline", args.baseline)
    before = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(before)
    torch_npu.npu.set_device(0)
    torch.manual_seed(219)
    results = []
    cases = [
        (1, 3, "default"),
        (4, 3, "default"),
        (10, 3, "default"),
        (40, 3, "default"),
        (40, 12, "default"),
        (10, 3, "forced_wide"),
        (40, 12, "forced_generic"),
        (10, 3, "wide_pool"),
    ]
    if args.only_wide_pool:
        cases = [case for case in cases if case[2] == "wide_pool"]
    for rows, heads, mode in cases:
        for module in [before, after]:
            module._FORCE_WIDE = mode == "forced_wide"
            module._CONFIG_OVERRIDE = (32, 2) if mode == "forced_generic" else None
        for support in [17, 2051]:
            q = torch.randn((rows, heads, 256), device="npu", dtype=torch.bfloat16)
            if mode == "wide_pool":
                # Real contiguous pools exceeding the int32 byte-span bound.
                # Initialize only rows selected by this benchmark, including
                # the final row whose byte offset exceeds INT32_MAX.
                k = torch.empty((4194305, 1, 256), device="npu", dtype=q.dtype)
                v = torch.empty_like(k)
                k[:128].copy_(torch.randn((128, 1, 256), device="npu", dtype=q.dtype))
                v[:128].copy_(torch.randn((128, 1, 256), device="npu", dtype=q.dtype))
                k[-1].copy_(torch.randn((1, 256), device="npu", dtype=q.dtype))
                v[-1].copy_(torch.randn((1, 256), device="npu", dtype=q.dtype))
            else:
                k = torch.randn((128, 1, 256), device="npu", dtype=q.dtype)
                v = torch.randn_like(k)
            slots = torch.full((rows, 2051), -1, device="npu", dtype=torch.int32)
            slots[:, :support] = torch.randint(
                1, 128, (rows, support), device="npu", dtype=torch.int32
            )
            if mode == "wide_pool":
                slots[:, support - 1] = k.shape[0] - 1
            k[0] = float("nan")
            v[0] = float("nan")
            inputs = q, k, v, slots
            if mode == "wide_pool":
                assert after.dispatch_info(*inputs)["offset_bits"] == 64
            expected = before.sparse_attention(*inputs)
            actual = after.sparse_attention(*inputs)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            timings = {"before_ms": [], "after_ms": []}
            methods = [("before_ms", before), ("after_ms", after)]
            for _, module in methods:
                for _ in range(3):
                    module.sparse_attention(*inputs)
            for repeat in range(args.repeats):
                order = methods if repeat % 2 == 0 else methods[::-1]
                for label, module in order:
                    torch_npu.npu.synchronize()
                    start = time.perf_counter()
                    for _ in range(args.iterations):
                        module.sparse_attention(*inputs)
                    torch_npu.npu.synchronize()
                    timings[label].append(
                        (time.perf_counter() - start) * 1000 / args.iterations
                    )
            row = {
                "rows": rows,
                "heads": heads,
                "width": 2051,
                "support": support,
                "mode": mode,
                "dispatch": after.dispatch_info(*inputs),
                "equal": True,
                "timings": timings,
                "medians_ms": {k: statistics.median(v) for k, v in timings.items()},
            }
            results.append(row)
            print(json.dumps(row), flush=True)
    args.output.write_text(
        json.dumps(
            {
                "device": torch_npu.npu.get_device_name(0),
                "baseline": str(args.baseline),
                "source": after.__file__,
                "results": results,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
