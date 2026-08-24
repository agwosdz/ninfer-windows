# Upstream feature evaluation, porting, and release workflow

This reference records the end-to-end workflow for borrowing a feature from upstream
(Neroued/ninfer or another fork/PR) into ninfer-windows and landing it on master.
It is the operating procedure used for the feat/decode-pr-fusions (upstream PRs #67
and #69) and debug/dflash2-accept-trace ports.

The workflow has four phases: evaluate, branch + worktree, port, and
land (merge -> sync -> cleanup). Keep the main checkout clean for interactive work;
all experimental edits happen in a worktree.

## 1. Evaluate the request

The user supplies an upstream commit link or a pull-request page. Before writing any
code, answer three questions:

1. What does the change actually do?
   - Fetch the patch. The sandbox/pwsh git path on this machine has TLS trouble
     (SEC_E_NO_CREDENTIALS), so fetch metadata and diffs through the GitHub API from
     inside the DSH runtime:
     fetch("https://api.github.com/repos/Neroued/ninfer/pulls/69",
       { headers: { "User-Agent": "dsh", "Accept": "application/vnd.github+json" }})
     gives the PR metadata; .../pulls/69/files lists changed files; the raw diff at
     https://github.com/Neroued/ninfer/pull/69.diff (or
     https://api.github.com/repos/<owner>/<repo>/commits/<sha> for a commit) gives the
     concrete patch to read.
   - Identify the files and the subsystem surface: which op, launcher, wrapper, family
     runtime file, or variant header does it touch?
2. Do we already have it?
   - Grep current origin/master for the marker symbols/signatures the change adds
     (git grep -n <symbol> origin/master -- <paths>), and check base/master (natpate
     fork) for a nearly-identical commit the fork already synced.
   - Distinguish identical features (nothing to do) from same-name-different-means
     (in-kernel tile prefetch vs cross-kernel L2 hint, etc.).
3. Is it feasible in this fork?
   - The fork has diverged from upstream: compressed-KV routes, DFlash2 drafter work,
     bench campaigns, MSVC/Windows porting. Verify each dependency the upstream change
     relies on exists in origin/master at the pre-change shape (member names, geometry
     constants, pdl primitives, workspace recipes, op signatures).
   - Read only the authorities that govern a live decision (AGENTS.md "Sources of
     truth"). A change that assumes a superseded path must be re-ported, not copied.
   - Judge against the product contract: identity set, single-GPU/single-model
     workload, bounded FIFO ingress. Features outside the contract are out of scope
     unless the user explicitly extends the contract.

Report one verdict: already have / not feasible / feasible + implement, with the
specific evidence (file + line markers, matching signatures). Also report the upstream
merge state (open PR vs merged) and that perf numbers measured on upstream Ubuntu/g++
must be re-measured on the Windows/MSVC build - they never transfer.

## 2. Branch and worktree

Local master is the user's active checkout and may hold in-progress edits. Never build
the feature there.

1. Create the branch from the current merge head:
   git fetch origin
   git checkout -b feat/<name> origin/master
2. Prefer a worktree so the main checkout stays untouched:
   git worktree add F:/Git/nf-agwosdz/worktrees/<name> <branch>
   (worktrees/ is gitignored at the repo root.)
   The main checkout at F:/Git/nf-agwosdz is reserved for the user's active edits; all
   feature commits are staged and committed inside the worktree.
   - If the workspace was moved (e.g. from a temp dir into worktrees/), re-verify every
     changed file survived and wipe any stale build/ cache before reconfiguring.

## 3. Implement: port, do not copy

- Port upstream changes to the current fork structure: same semantics, adapted to the
  fork's compressed-KV / DFlash / 27B-35B variant layout. Do not blindly cherry-pick.
- When multiple upstream PRs touch the same kernel/surface with independent effects
  (e.g. PR #67 folds the MoE router selection into D1 and PR #69 warms L2 from D1's
  idle window), merge them deliberately into one edit and keep the combined commit
  (one perf(...)/feat(...) commit naming both). Never split a combined edit into two
  cherry-picks.
- Thread new cross-layer or cross-function parameters explicitly through arguments
  (e.g. WeightPrefetchSpan through run_layers -> mlp_tail -> Variant::post_mixer).
  Per AGENTS.md, no ambient/global mutable state.

### Edit hygiene

- Read files with the read tool first: edit/write enforce a read-first policy per path.
- Match the exact whitespace present in the fork; continuation lines in this codebase
  are often 26-space indented. When an easy edit fails on "old_string not found"
  repeatedly, re-read the exact text and build the replacement from the read output
  instead of retyping.
- Multi-line replacement blocks are dangerous: an earlier variant.h restore replaced a
  10-line block and silently dropped four unrelated using/constexpr lines. After every
  multi-line edit, re-read the whole region and git diff --stat + git diff --check to
  confirm no accidental truncation.
- Do not leave local scratch .bat configure helpers in the tree; write them to %TEMP%
  or remove them before committing.

## 4. Build and verify on MSVC

- ninfer builds are heavy CUDA compiles; use the VS dev environment so nvcc can find
  cl.exe:
  call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
  cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_TOOLCHAIN_FILE=C:/src/vcpkg/scripts/buildsystems/vcpkg.cmake ^
    -DVCPKG_TARGET_TRIPLET=x64-windows ^
    -DCMAKE_CUDA_ARCHITECTURES=120a   [ -DBUILD_TESTING=ON ]
  cmake --build build -j   (unrestricted parallelism)
- In an automated/pwsh shell, write a small .bat that calls vcvars64.bat then cmake or
  ninja and invoke it via cmd /c; nested "cmd /c call ... && ..." in PowerShell mangles
  the spaces in "Program Files (x86)".
- Build targets: ninfer_core, ninfer_ops, then ninfer.exe / ninfer-serve.exe; for tests
  build the ninfer_*_test targets.
- Real-model tests live in the main checkout's models/; they read the artifact from env
  vars set in the test process:
  - NINFER_QWEN3_6_35B_A3B_WEIGHTS=.../qwen3_6_35b_a3b.ninfer
  - NINFER_QWEN3_6_27B_WEIGHTS=.../qwen3_6_27b.ninfer
- Focused verification for a decode/op change:
  - op-level: ninfer_sparse_moe_test, ninfer_gqa_attention_test, ninfer_rope_test
  - real-model: ninfer_qwen3_6_35b_a3b_real_test (MoE decode with fused kernels) and
    ninfer_qwen3_6_27b_prefix_real_test (dense qk-norm/rope + gate fusion)
  - git diff --check must pass.
- Perf numbers are NOT inherited from upstream. Decode-perf claims (e.g. PR #67's
  +2.3% from three decode node fusions, PR #69's +3.7% L2-warmup gain) were measured on
  Ubuntu/g++ and must be re-measured on the Windows/MSVC build before report; document
  the deviation if the measurement is deferred.

## 5. Commit, push, merge, sync

- Commit with a Conventional Commit subject in the repo's lowercase style
  (feat(...) / perf(...) / fix(...) / chore(...)). Write the body to a temp file and
  git commit -F, or quote carefully - PowerShell mangles multi-line -m.
- Commit only the feature files; leave the user's unrelated in-progress edits unstage.
- Push with -u; GitHub prints the PR link (/pull/new/<branch>).
- After the user merges the PR to master on GitHub:
  - git fetch origin (when local git fails on TLS, use the GitHub API fetch inside the
    runtime instead of --network)
  - Discard local copies of any file that is now byte-identical to origin/master (e.g.
    an uncommitted edit the merged PR already includes): git checkout -- <file>
  - Fast-forward local master: git merge --ff-only origin/master
- Cleanup: git worktree remove <path>, git branch -d <merged-branch>, and/or delete the
  remote branch once the PR is merged, per the user's preference.

## Task board as the standing driver

This repository is worked through the DSH **task board** (sidebar "任务看板", the
installed dsh-task-board plugin). Every piece of planned repo work that comes out of
this workflow - an upstream PR to evaluate, a port to implement, a bench to run - is
created as a board task rather than only handled in a chat turn.

Defaults for repo tasks:

- **Scope**: every planned piece of repo work becomes a board task; quick interactive
  turns stay in chat, but anything structured (evaluate -> port -> verify -> PR) is a
  task.
- **Pinning**: tasks are pinned to this workspace (the repo checkout) and the repo agent
  preset, so the Host executes them with the same checkout, permissions, and
  environment as this session - including after the browser is closed.
- **Execution**: the Host runs and settles the task; API quota is consumed by execution.
  The board survives browser close; only the computer being asleep/hibernating or shut
  down stops it (idle sleep protection is off by default and does not cover lid close,
  manual sleep, hibernate, or wake-from-sleep).
- **Cron**: recurring or "catch up" work (e.g. a periodic upstream-master sync scan) can
  use the board's 5-field cron in the Host local timezone; missed triggers do not
  backfill.

When a task lands on the board for this repo:

1. **Evaluate first** (section 1) - fetch the upstream ref via the DSH runtime GitHub
   API, produce the "already have / not feasible / feasible" verdict, and the merged
   edit plan.
2. **Port on a worktree** (sections 2-4) inside the same repo checkout; the browser may
   close once the task is running - the Host settles it with the pinned workspace.
3. **Land** (section 5) - commit, push the feat branch, and open/merge the PR; then
   sync local master and clean up the worktree and branch.

## Recurring local habits

- Keep bin/ and worktrees/ (and the session helper .bat files) out of git; they're
  gitignored at the repo root.
- Never pick an artifact by glob or "latest"; bind real-model tests to the exact path
  under ..\models\.
