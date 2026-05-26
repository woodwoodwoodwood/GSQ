"""Install a sitecustomize-style fix into the active site-packages.

Wraps humming.ops.humming_gemm so that, when vLLM passes a garbage
`expert_layout` (uninitialized int64 buffer during profile_run on H20),
we replace it with a synthesized balanced layout so the kernel can run
to completion. This unblocks profile_run; real inference still uses the
true expert_layout produced by the router.

Usage on the box:
    python /home/cakejiang/github/GSQ/scripts/install_humming_fix.py
"""

import os
import site
import sys

SITE = "/data/miniconda3/envs/env-3.12/lib/python3.12/site-packages"
TARGET = os.path.join(SITE, "usercustomize.py")

PAYLOAD = '''import os
try:
    import torch
    from humming import ops as _hops
    _orig = _hops.humming_gemm

    def _wrap(layer_config, compute_config, tuning_config,
              inputs, weight,
              outputs=None, input_scale=None, weight_scale=None,
              zero_point=None, bias=None, global_scale=None,
              sorted_ids=None, expert_ids=None, num_tokens_padded=None,
              expert_layout=None, locks=None,
              top_k=1, valid_shape_m=0):
        # Sanitize garbage expert_layout from vLLM profile_run dummy buffers.
        if expert_layout is not None and expert_layout.dtype == torch.int64:
            max_valid = inputs.shape[0]
            bad = (expert_layout < 0) | (expert_layout > max_valid)
            diffs = expert_layout[1:] - expert_layout[:-1]
            non_monotonic = bool((diffs < 0).any().item())
            if bool(bad.any().item()) or non_monotonic:
                num_experts = expert_layout.numel() - 1
                step = max_valid // max(num_experts, 1)
                new_layout = torch.arange(num_experts + 1, dtype=torch.int64,
                                          device=expert_layout.device) * step
                new_layout[-1] = max_valid
                print(f"[FIX-EL] sanitized expert_layout num_exp={num_experts} "
                      f"max_valid={max_valid} step={step}", flush=True)
                expert_layout = new_layout
        return _orig(layer_config, compute_config, tuning_config,
                     inputs, weight, outputs, input_scale, weight_scale,
                     zero_point, bias, global_scale,
                     sorted_ids, expert_ids, num_tokens_padded,
                     expert_layout, locks, top_k, valid_shape_m)

    _hops.humming_gemm = _wrap
    print("[FIX-EL] hook installed (sanitize expert_layout)", flush=True)
except Exception as e:
    import traceback
    print(f"[FIX-EL] init err: {e}", flush=True)
    traceback.print_exc()
'''

with open(TARGET, "w") as f:
    f.write(PAYLOAD)
print(f"Wrote {TARGET} ({len(PAYLOAD)} bytes)")
