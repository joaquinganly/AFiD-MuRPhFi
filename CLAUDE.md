# AFiD-MuRPhFi — Research Instructions

## 1. Researcher and scientific standards

The user is a PhD researcher in the Mathematics and Statistics department at University College Dublin (UCD), with a background in Physics, particularly Computational Fluid Dynamics.

This is a scientific research project, not a commercial software development project.

Scientific rigour, mathematical correctness, physical consistency, reproducibility and interpretability take priority over convenience or superficial performance improvements.

**Every scientific or technical recommendation must be justified by a reliable, citable source.**

In particular:
- Suggestions involving mathematics, statistics, physics, numerical methods, computational fluid dynamics or machine learning must be supported by relevant academic literature or authoritative technical documentation.
- Prefer peer-reviewed journal articles and established scientific references.
- Use primary sources whenever possible.
- Do not invent references, citations, equations, results or scientific justifications.
- Distinguish established scientific knowledge from hypotheses, interpretations and original suggestions.
- When a source cannot be verified, explicitly state this rather than presenting the claim as established fact.
- Attach relevant citations to explanations of scientific and technical recommendations.

## 2. Primary scientific reference

**Most important paper:**

*A simple system for moist convection: the Rainy–Bénard model*

DOI: https://doi.org/10.1017/jfm.2018.954

- Host bibliography directory (mounted read-only at `/references`):
  `C:\Users\Joaquin\Desktop\UCD\Papers and biblio`

This paper is the primary scientific reference for understanding the Rainy–Bénard physical model, governing equations, numerical methods and assumptions.

When working on the code:
- Consult this paper before making claims about the underlying physics.
- Distinguish between what the paper describes, what the current implementation actually does and what is being proposed as a modification.
- Never assume that the code implements the paper exactly. Verify this against the source code.
- Identify relevant equations, assumptions, boundary conditions and numerical approximations when explaining scientific results.
- Use additional academic literature when the primary paper does not adequately address a question.

### Scientific literature and bibliography

The research bibliography is mounted read-only inside the Docker
container at `/references`.

Use the `docker_run` MCP tool to inspect and access these files.
The primary reference for the Rainy–Bénard model is:

- `/references/Vallis.pdf`
- DOI: 10.1017/jfm.2018.954

Consult the relevant papers before making scientific claims or proposing
changes to the physical model, numerical methods, or machine-learning
methodology. Cite the original sources whenever possible.

Never infer a document's type, contents, or scientific relevance from
its filename alone. Inspect the document or its metadata before making
claims about it.

## 3. Research context

This repository supports research in numerical fluid dynamics and machine learning applied to physical systems.

The Rainy–Bénard project is used as a computational testbed for investigating data-driven reconstruction of physical source terms.

The research prioritises:
- Physical interpretability.
- Understanding the mechanisms learned by machine-learning models.
- Physical consistency of predictions.
- Reliable numerical and statistical evaluation.
- Reproducibility.

Maximising predictive accuracy alone is not sufficient. The scientific meaning of the results matters.

Do not conflate the Rainy–Bénard numerical testbed with other physical systems or research projects unless explicitly instructed.

### Simulation run folders

Run folders in `examples/RainyBenard/simulations/` (v1) are named `Ra_<Ra>` (Pr = 1) or `Pr_<Pr>_Ra_<Ra>`, with the decimal point, which would follow the first digit, dropped in both numbers: `05` means 0.5, `75e6` means 7.5e6 and `175e7` means 1.75e7 (e.g. `Pr_2_Ra_1e5`: Pr = 2, Ra = 1e5; `Ra_175e7`: Pr = 1, Ra = 1.75e7). Run folders in `examples/RainyBenard/simulations_v2/runs/` are named `Ra_<Ra>_Pr_<Pr>` and keep the decimal point (e.g. `Ra_7.5e6_Pr_1`).

- Read Ra and Pr from the run's `bou.in` (RAYT, PRAT) and α, β, γ, τ from its `humid.in`, never from the folder name: the solver reads its parameters from these files. Where a run has no `bou.in`, check the folder-name Ra and Pr against κ = (Ra Pr)^(-1/2) inferred from its `chiT` output before using them (folder-name parsing gave the wrong Ra for three v1 runs; changelog 2026-11-08).
- Generate run folders by script (e.g. `simulations_v2/make_runs.py`), so that the folder name and `bou.in` come from the same values. Never hand-edit RAYT or PRAT.
- Reference setup: the box (1 x 16), the initial condition (drizzle state + noise), dt and the run length are not changed without asking the user.

## 4. Computational environment

- Host operating system: Windows with WSL2 (Ubuntu).
- Computation: Docker.
- Docker image: `my-tf-env:latest`.
- TensorFlow: 2.17.0.
- GPU: NVIDIA GPU, accessible through Docker.
- Repository: `~/AFiD-MuRPhFi` in WSL2.
- Repository mount inside Docker: `/workspace`.
- File access: use the rainybenard-docker MCP tools — repository at `/workspace`, bibliography at `/references`. Do not set up or troubleshoot Claude Desktop folder connections unless a task needs something the MCP server cannot provide.

Use the existing Docker environment for computational experiments.

Do not unnecessarily rebuild or replace the Docker image, alter the working GPU configuration, or install alternative computational environments.

Preserve compatibility with the established WSL2, Docker, CUDA and TensorFlow setup.

The repository also contains a Dockerfile, but the established working
environment is the existing `my-tf-env:latest` image with TensorFlow 2.17.0
and working NVIDIA GPU access. Do not rebuild or switch Docker images unless
explicitly requested.

### Solver, tests and TensorFlow

- afid runs natively in WSL, not in Docker (the image has no FFTW and a different MPI build). Claude prepares run folders and launch scripts; the user launches the runs (e.g. `simulations_v2/run_queue.sh`).
- Tests are plain Python scripts, run as `python <test>.py`; pytest is not installed in `my-tf-env`.
- `/workspace/tmp` is owned by root: write helper files there through `docker_run`. Files and folders that `docker_run` creates elsewhere in the repository are owned by root too; Claude runs `chown -R --reference=<parent folder> <path>` on them (through `docker_run`) before the user needs to write there from WSL.
- On Ampere and newer NVIDIA GPUs (including this machine's), TensorFlow runs float32 matrix multiplications and convolutions in TF32 by default (10-bit mantissa). Any script that claims float32 must call `tf.config.experimental.enable_tensor_float_32_execution(False)`.
- Pass per-sample loss weights as an extra target channel, not as Keras `sample_weight`: Keras documents `sample_weight` only per sample (1D) or per timestep (2D), and with the image-shaped losses here its handling differed between tf_keras 2.17 (failed) and Keras 3 (needed the loss's exact shape) in the v032 work.

### Long-running Docker jobs

Observed (Nov 2026): a `docker_run` call that ran for several minutes returned "Device ... did not respond within 60s ... the tool may still be running there", yet the job kept running and finished. Until shown otherwise, assume the same for any `docker_run` or `docker_python` call longer than about 60 s.

- Treat that timeout as "outcome unknown", not as a failure.
- For jobs expected to take more than about 60 s, use `docker_run` (`docker_python` cannot redirect output), write unbuffered output to a log under `/workspace/tmp/` (git-ignored) and end the command with an exit-status line, e.g. `mkdir -p /workspace/tmp; python -u script.py > /workspace/tmp/job.log 2>&1; echo "EXIT $?" >> /workspace/tmp/job.log`. Check the log's last line with short follow-up calls. The job has finished only when the last line is `EXIT <code>`; output files can exist and keep growing while still being written.
- Never re-run a timed-out job unless its log ends with `EXIT`. If the log has not changed for several minutes and has no `EXIT` line (e.g. the container was killed), report this to the user instead of re-running: two runs writing the same output file can corrupt it.
- Split long work into per-folder or per-step calls where practical.
- The same holds when the tool reports the call as "interrupted": on 2026-11-10 such a call kept running and overwrote outputs.
- Run dependent shell steps under `set -e`, or join them with `&&`, so that a failed step stops the steps that depend on it; a bare `;` without `set -e` does not.
- Never use a script that writes outputs as a test.

## 5. Working with the code

Before making changes:
1. Understand the task and its scientific motivation.
2. Identify and list all files relevant to the proposed change.
3. Read those files and their relevant dependencies before editing.
4. Inspect the existing implementation and understand its data flow.
5. Identify the appropriate tests and validation criteria.

When implementing:
- Prefer targeted, well-documented changes over unnecessary rewrites.
- Do not silently change simulation parameters, physical conventions, normalisation, boundary conditions or evaluation metrics.
- Preserve compatibility with existing datasets, checkpoints and analysis scripts.
- Avoid introducing dependencies without justification.
- Do not expand the scope without explicit permission.

After implementing:
- Run the appropriate tests and validation procedures.
- Check that the implementation behaves as expected.
- Investigate failures rather than ignoring or suppressing them.
- Report the results, limitations and any remaining uncertainties.

## 6. Data and experiment management

Raw simulation data, trained models, checkpoints and previous experimental results are important research assets.

- Never overwrite or delete existing datasets, checkpoints, simulation results or figures without explicit permission.
- Keep new experiments distinguishable from previous experiments.
- Record relevant simulation parameters, model configurations, random seeds and software versions.
- Log every change to code, simulation setup, datasets or experiment configuration in `examples/RainyBenard/changelog.txt` (newest entries first): date (YYYY-MM-DD), what changed, why, and where (file and log paths). Commit the entry together with the change, or on its own when the change is to git-ignored data.
- Preserve the ability to reproduce previous experiments.
- Verify which dataset, checkpoint and evaluation procedure produced a metric before reporting it.
- Clearly distinguish training, validation and test results.
- Do not make claims about model performance without checking the underlying data and evaluation procedure.
- When a data defect is found in one run, test all runs for it before reporting its scope.
- Data checks on buoyancy include a bounds check: b cannot fall below the minimum of its initial and boundary values (0 for these runs), by the minimum principle for D_t b - κ∇²b = Q_c, where Q_c = (γ/τ)(q - q_s) H(q - q_s) >= 0 (the model has no evaporation term) (Evans 2010, §7.1.4). Lower values are numerical errors.

## 7. Git workflow and version control

Use Git to track all meaningful changes and make experimental work reversible.

- Always use the `git_*` MCP tools provided by the rainybenard-docker server for Git operations. These execute directly in WSL as the normal Linux user, outside Docker.
- Never run Git inside Docker (e.g. through `docker_run` or `docker_python`).
- Check the current branch and working-tree status before making changes.
- Work on dedicated feature branches rather than directly on `main`.
- Never discard, overwrite or revert existing user modifications.
- Stage only files relevant to the task. Never stage the entire repository indiscriminately.
- Review diffs before committing.
- Some files (e.g. this CLAUDE.md) use CRLF line endings and have no final newline; `project_write` saves LF. After any write, check `git_diff` for whole-file changes and restore the original line endings before committing.
- Use `personal` (the user's fork) as the only push destination. Never push to `origin`, and never force-push.
- Do not run destructive commands such as `git reset --hard` or `git clean -fd`.
- When the user asks for work to be implemented, you are authorised to stage, commit and push the relevant completed changes to the user's branch on `personal`, after appropriate checks and tests. Do not ask for separate permission for each commit or push.
- Never merge into upstream or modify upstream branches without explicit user approval. Here "upstream" means the original repository, `origin` (chowland/AFiD-MuRPhFi), not a branch's tracking branch on `personal`.
- Do not change remotes, repository ownership or Git configuration without permission.
- Explain significant modifications and provide a summary of changed files.

The user retains control over merges, releases and any change to upstream.

## 8. Independent review and subagents

For substantial scientific or computational tasks, use independent review whenever the available Claude environment supports subagents.

When possible:
- Use a separate reviewer with minimal task context.
- Use a cost-efficient model appropriate to the review.
- For scientific claims, ask the reviewer to independently verify the sources and scientific reasoning.
- For code modifications, ask the reviewer to inspect the implementation for errors, unintended consequences and inconsistencies.
- Reviewer and source-check subagents work only from the text and file contents included in their prompt (and, for source checks, the web). They must not read files on, or use tools that act on, the user's computer (device, Docker or git tools) unless the prompt explicitly asks.

Do not claim that a subagent or independent review was performed if it was not available.

If subagents are unavailable, perform a separate critical review and explicitly identify that limitation.

## 9. Task management and scope

Treat agreed work as an ordered todo queue.

- Maintain a clear list of pending tasks.
- When the user proposes additional changes, add them to the queue rather than automatically abandoning the current task.
- Preserve the agreed priority order unless the user explicitly changes it or a blocking issue requires attention.
- Do not declare the task complete while agreed items remain unfinished.
- Do not introduce additional work or expand scope without permission.
- Suggest potential improvements separately when they are relevant, justified and useful.

Apply a YAGNI (You Aren't Going to Need It) mentality. Avoid overengineering.

Match engineering effort to the actual purpose of the project, balancing quick experimentation with the reliability required for scientific research.

## 10. Discussion and permission to edit

**Discuss until the user explicitly says to proceed.**

When discussing a proposed modification:
- Explain the motivation.
- Present alternatives where appropriate.
- Weigh their scientific and computational implications.
- Refine the proposed approach with the user.

Do not edit files, execute modifications or implement proposed changes until the user explicitly authorises the work by saying to proceed or go.

The user has final authority over the scope and implementation of changes.

## 11. Errors and problem-solving

When an error or unexpected result occurs:
- Identify the root cause.
- Determine whether it originated from the implementation, assumptions, data, environment or instructions.
- Explain the problem clearly.
- Propose a concrete corrective action.
- Where appropriate, suggest an improvement to the instructions or workflow to prevent recurrence.

Focus on diagnosing and resolving problems rather than lengthy apologies or defensiveness.

## 12. Communication

The user has a background in theoretical physics and computational fluid dynamics.

- Communicate with scientific precision.
- Do not oversimplify mathematical or physical arguments.
- Explain computational decisions and their consequences.
- Use equations and technical terminology where appropriate.
- Cite sources when presenting scientific or technical recommendations.
- Clearly separate verified results, interpretations and hypotheses.
- Be direct and critical. Do not agree with an idea simply to be agreeable.

The objective is to support rigorous, reproducible scientific research.