#!/usr/bin/env python3
"""Reference values for src/omnigen2/sched.rs from the official OmniGen2 scheduler.

    python tools/ref_omnigen2_sched.py /path/to/OmniGen2 [steps] [num_tokens]

`/path/to/OmniGen2` is a checkout of github.com/VectorSpaceLab/OmniGen2. Runs
`DPMSolverMultistepScheduler(algorithm_type="dpmsolver++", solver_type="midpoint",
solver_order=2, prediction_type="flow_prediction")` (the `--scheduler dpmsolver++` of the
official inference.py) on the sample [0.5, -1, 0.25, 2] with the model outputs
`((7 i + 3 k) % 11) / 8 - 0.5` (step i, element k), and prints the sigmas and the sample after
each step as the constants of the Rust test. Needs torch, numpy, scipy and diffusers; no weights.
"""
import importlib.util
import sys

import torch

repo = sys.argv[1]
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 6
num_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 256

# The scheduler file alone: importing the package would pull in the whole pipeline.
spec = importlib.util.spec_from_file_location(
    "og2_dpm", f"{repo}/omnigen2/schedulers/scheduling_dpmsolver_multistep.py"
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

sched = mod.DPMSolverMultistepScheduler(
    algorithm_type="dpmsolver++",
    solver_type="midpoint",
    solver_order=2,
    prediction_type="flow_prediction",
)
sched.set_timesteps(steps, num_tokens=num_tokens)


def f32(v):
    return ", ".join(repr(float(x)) for x in v)


print(f"    // timesteps: [{f32(sched.timesteps)}]")
print(f"    const REF_SIGMAS: [f32; {steps + 1}] = [{f32(sched.sigmas)}];")
x = torch.tensor([[0.5, -1.0, 0.25, 2.0]], dtype=torch.float32)
rows = []
for i, t in enumerate(sched.timesteps):
    v = torch.tensor([[((i * 7 + k * 3) % 11) / 8.0 - 0.5 for k in range(4)]], dtype=torch.float32)
    x = sched.step(v, t, x, return_dict=False)[0]
    rows.append(f"        [{f32(x[0])}],")
print(f"    const REF_SAMPLES: [[f32; 4]; {steps}] = [")
print("\n".join(rows))
print("    ];")
