# Contributing

Help learners understand an update, reproduce an experiment, or diagnose a failure.
Corrections and focused pull requests are welcome. Keep the course's 40-topic structure;
prefer a deeper example or a clearer connection over another algorithm heading.

For an issue, include the notebook/topic, expected behavior, actual result, and enough
information to reproduce it. For notebook failures, include Python/package versions,
seed, full error text, and whether you restarted the kernel and ran every cell.

For code changes, run the affected notebook from a fresh kernel. Keep saved outputs
readable, report failures as well as successes, and record any changed seeds or budgets.
Add checks for substantive mistakes such as terminal masking, target gradients, or
evaluation leakage. Avoid tests that merely repeat the implementation.

For algorithm comparisons, use the same task and state clearly whether interaction,
compute, or both are controlled. Separate prediction from control, training behavior
from frozen-policy evaluation, and proxy reward from independently verified success.
Do not discard unsuccessful seeds or describe a toy task as a general benchmark.

For explanations, connect intuition → update → experiment → limitation. Prefer original
papers or official documentation near the relevant claim. Keep new visuals consistent
with ink-plum, antique gold, sage, muted plum, and warm paper. Preserve readable alt text.
README equations use rendered SVGs to avoid the renderer errors encountered earlier.

Keep the main README as the quick-read course. Put deeper derivations and original
worked exercises in the corresponding [chapter](chapters/README.md), preserving its
ten numbered sections and navigation. Follow the [notation guide](chapters/NOTATION.md),
state assumptions before guarantees, and distinguish implemented labs from proposed
extensions. Check local links and numerical examples when editing a chapter; prose-only
changes do not require rerunning unrelated notebooks.

Authoring scripts, local model caches, editor files, credentials, and virtual environments
should stay local. Learner-facing code in `rl_course/` belongs in the repository because
the integrated notebooks import it. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for setup.

Contributions to code use [MIT](LICENSE-MIT). Educational prose and visual assets use
[CC BY 4.0](LICENSE-CC-BY-4.0). Identify any third-party material and retain its notices;
model licenses are separate from course licenses.
