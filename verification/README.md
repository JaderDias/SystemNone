# Formal verification (TLA+)

Four TLA+ models of the concurrent parts of the gateway and the worker, checked with the TLC
model checker. They found seven bugs, fixed in the same change (see [Bugs](#bugs)). Each model has
a `Fixed` constant: `TRUE` models the code as it is now, `FALSE` the code before the fixes, so
every counterexample can be replayed.

## Running

```bash
verification/check.sh              # every model as the code is: all properties hold
verification/check.sh --original   # the code before the fixes: every bug shows up again
verification/check.sh --limits     # what the fixes leave open: a worker that dies mid-job
```

`check.sh` uses `tlc` from `PATH`, or `TLA2TOOLS=/path/to/tla2tools.jar` with Java 11 or later
([TLC 1.7.4](https://github.com/tlaplus/tlaplus/releases/tag/v1.7.4)), or else Nix: it re-runs
itself inside `nix shell nixpkgs#tlaplus`. On Windows, run it in WSL. `make verify` runs the
first two, and so does CI, on every push and pull request: a property that fails, or a bug that
no longer shows up before its fix, fails the build.

The models are written in PlusCal. After changing an algorithm, translate it again and commit the
result: `pcal -nocfg JobLifecycle.tla` (in `nix shell nixpkgs#tlaplus`), or
`java -cp tla2tools.jar pcal.trans -nocfg JobLifecycle.tla`.

## How the models follow the code

- The gateway runs on tokio's multi-threaded runtime, so its handlers run in parallel and can
  interleave anywhere between two shared-state operations, not only at `.await`. Each label is
  one such step: one SQLite call (each `Db` method locks the connection on its own), one write
  or read of an in-memory lock, or one store call.
- Comments give the function and the lines it had in `main` at ffc396a, before this change. The
  retries (#58) and the admin's Retry (#63) came later; their labels name the functions only.
- Liveness uses weak fairness: a handler that has started finishes, but a request may never come
  (labels marked `:-`). The main task of `InputLog` gets strong fairness, as tokio's `Mutex` is
  first come, first served.
- The bounds are small (two workers, two jobs, three inputs, ...), so each run takes seconds.
  They are enough for every counterexample below.

## The models

### JobLifecycle: jobs and the students they deploy

One prompt's jobs from the queue to deployment: `count_request` and `requeue_ready` (the threshold),
"Train now", "Stop using student", cancel and Retry from the admin, and two workers that claim,
report progress, train, publish and complete, with two gateway tasks serving `/complete`, so
repeated calls can run at once. A `/complete` call can be lost in transit (connection refused, or
a timeout), and with `WorkerCrash` a worker can die mid-job. A failure the worker reports is
retried until `MaxAttempts` attempts have failed: the retry is a new queued job, one of the
`MaxJobs` jobs the model may create. The admin's Retry of a job an admin cancelled is one too,
at attempt 1.

Abstracted: what training does (it publishes the student or it doesn't), the job spec,
`job_spec` failing at claim time (the same shape as a student that can't be loaded: no retry),
other prompts (they share only `queue_lock` and the claim order), gateway restarts, and time: a
claim may take a retry at any point after it is queued, which includes every point after its
`not_before`. `claim_job`'s time check only removes behaviours, so every safety property holds
with it too, and "Train now" making a waiting retry claimable at once changes nothing the model
tracks. That the check holds a retry back is a Rust test
(`db::tests::claims_leave_a_retry_until_its_time`). A Retry of a job no admin cancelled gets a 409
and changes nothing, so the admin only picks cancelled jobs. With two jobs, an admin's Retry
never has a job left for an automatic retry after it; that its failure starts the backoff over is
a Rust test (`an_admin_retries_a_cancelled_job`).

| Property | Kind | Meaning |
|---|---|---|
| `AtMostOneOpenJob` | invariant | at most one queued or running job per prompt |
| `ClaimedOnce` | invariant | no job is handed to two workers |
| `ServedIsDone` | invariant | the student `/v1/classify` uses is from a job that completed successfully |
| `CancelledNeverDeploys` | invariant | a job an admin cancelled never deploys its student |
| `ThresholdOnlyWhileCollecting` | invariant | the request threshold queues a job only for a prompt that is collecting |
| `MemoryMatchesDb` | invariant | between requests, the prompt's status and student in memory are SQLite's |
| `StatusHasJob` | invariant | between requests, a "queued" or "training" prompt has a job in that state |
| `RetriesFollowWorkerFailures` | invariant | an automatic retry follows one failure the worker reported, as its next attempt |
| `CancelledNeverRetried` | invariant | a job an admin cancelled is never retried automatically |
| `AdminRetriesFollowCancels` | invariant | a job the admin's Retry queued follows a job an admin cancelled, which stays cancelled, and starts over at attempt 1 |
| `AttemptsBounded` | invariant | no training gets more than `MaxAttempts` attempts |
| `RetriedUntilTheLastAttempt` | invariant | between requests, every failure the worker reported before the last attempt has its retry |
| `FinishedJobsStay` | action | a finished job's status and message never change |
| `ClaimedJobsFinish` | liveness | every claimed job eventually finishes |
| `QueuedJobsLeaveTheQueue` | liveness | every queued job is eventually claimed or cancelled |
| `LockReleased` | liveness | `queue_lock` is always released |

### InputLog: the buffered input log

Three requests over two prompts push inputs into the buffer; the background task flushes (on its
timer or a full batch), a claim flushes before it hands out a job, and main flushes once more
after the server stops, then returns, which drops the background task wherever it is. One store
write may fail, and its records go back to the buffer.

Abstracted: the text of the records, the `pending` counter and the wake-up (they only decide
when the background task flushes), and a store that fails during shutdown (then the records are
lost: there is no local spool).

| Property | Kind | Meaning |
|---|---|---|
| `NoDuplicates` | invariant | no input is written twice |
| `ClaimSeesEarlierInputs` | invariant | once a claim's flush returns, the store holds every input logged before the claim, unless a write failed meanwhile ("the worker must see every logged input") |
| `NoLossOnShutdown` | invariant | once main returns, every logged input is in the store |
| `InputsEventuallyWritten` | liveness | every logged input is eventually written |

### Serving: hot swaps and student downloads

Two requests classify while the deployed student is swapped and removed three times (complete
and undeploy). Separately, two `load_student` calls for the same job (a repeated `/complete`
call, or a restart, which loads every active deployment) download its three files into the
cache, with one failed store read.

Abstracted: the prediction itself, the threshold (it travels in the same `Arc<Deployed>` as the
student), and crashes mid-download (a failed read leaves the same partial folder).

| Property | Kind | Meaning |
|---|---|---|
| `InFlightStudentAlive` | invariant | a request never predicts with a student that was dropped |
| `NoLeak` | invariant | a student is dropped once neither the prompt nor a request uses it |
| `LoadFailsOnlyOnItsOwnError` | invariant | a load fails only because of its own store error, never because of what another load left in the cache |

### AdminConfig: settings, System One models and overrides

Three admin requests at once, each saving the settings, a System One model or a prompt's
overrides. Concurrent saves of the settings are last-writer-wins: the admin page always sends
every field.

| Property | Kind | Meaning |
|---|---|---|
| `CacheMatchesDb` | invariant | between requests, the cached settings and System One models are SQLite's |
| `OverridesMatchDb` | invariant | between requests, a prompt's overrides in memory are SQLite's |

## Results

TLC 2.19 (tla2tools 1.7.4), 16 cores:

| Model | Distinct states | Time | Code now | Code before the fixes |
|---|---|---|---|---|
| JobLifecycle | 925,515 | 72 s | all hold | `CancelledNeverDeploys`, `ServedIsDone`, `FinishedJobsStay`, `MemoryMatchesDb`, `StatusHasJob`, `ThresholdOnlyWhileCollecting`, `ClaimedJobsFinish` violated |
| InputLog | 105,204 | 6 s | all hold | `ClaimSeesEarlierInputs`, `NoLossOnShutdown` violated |
| Serving | 729,756 | 5 s | all hold | `LoadFailsOnlyOnItsOwnError` violated |
| AdminConfig | 9,838 | < 1 s | all hold | `CacheMatchesDb`, `OverridesMatchDb` violated |

JobLifecycle has modelled retries since #58, with two attempts per training; before them it had
407,942 states (28 s). The admin's Retry (#63) took it from 867,818 states (64 s) to 925,515.

`AtMostOneOpenJob`, `ClaimedOnce` and `NoDuplicates` held before the fixes too (11 million
states for JobLifecycle then), as did the hot-swap properties: `queue_lock`, the claim
transaction, `mem::take` and the `Arc` already did their jobs. With `WorkerCrash`
(`check.sh --limits`), `ClaimedJobsFinish` fails even now; see below.

## Bugs

Each has a regression test that fails before its fix.

1. **A cancelled job could deploy.** `complete_job` checked that the job was running, awaited
   `load_student`, then deployed and marked it done without checking again; `finish_job`
   overwrote any status. An admin cancelling in that window got a 200, and the student deployed
   anyway, with the job "done"; cancelling just after the completion marked a serving student's
   job "failed". Now `finish_job` only finishes a queued or running job and says whether it did,
   and the student deploys only if its completion is the one that finished the job. Test:
   `db::tests::finished_jobs_stay_finished`.
2. **Progress after a cancel replaced the reason.** The worker isn't told of a cancel, and its
   next progress report overwrote "cancelled by an admin". Progress now only updates a running
   job, and gets a 409 otherwise. Test: `a_cancelled_job_stays_cancelled`.
3. **Memory and SQLite could disagree on a prompt.** `queue_lock` only covered queueing; claims,
   cancels, completions and undeploys wrote the status and the student to SQLite and to memory
   in separate steps, and two of them could interleave: a prompt could be "queued" in SQLite and
   "training" in memory, or serve a student its deployment row marks inactive. They now all run
   under `queue_lock`. Covered by the model (the windows are between two synchronous calls, so no
   test can hit them reliably).
4. **The threshold could queue a job for a prompt that wasn't collecting.** `count_request` and
   `requeue_ready` check the status before taking `queue_lock`; a "Train now" and a cancel in
   between left no open job, and a new one was queued right after the cancel. `queue_job` checks
   again under the lock. Test: `the_threshold_queues_only_collecting_prompts`.
5. **A flush could return before the inputs were written.** The background task takes the
   buffer, then writes it with one `.await` per prompt; a claim's flush meanwhile found the buffer
   empty and returned, so the worker trained without those inputs, and at shutdown main returned
   and the runtime dropped the background task with the records it held. Flushes now take turns.
   Test: `store::tests::a_flush_returns_once_the_flush_in_progress_is_written`.
6. **A partial download broke a student for good.** `load_student` skips the download when
   `model.onnx` exists, and the download wrote the files in place: after a failed read, or while
   another download of the same job was writing, the next load opened a partial folder, and kept
   failing. Files now go to a temporary folder, renamed into place once complete. Test:
   `store::tests::a_failed_download_leaves_no_partial_student`.
7. **The worker dropped a finished job when the gateway was restarting.** An
   `httpx.TransportError` on `/complete` ended `run_worker`, and the job stayed running for good.
   The worker now keeps the result and tries again, as it does for claims; with the fixes above,
   a repeated call is safe. Test: `test_worker_keeps_its_result_until_the_gateway_takes_it`.

Two smaller races in the admin configuration, fixed the same way: `reload_config` could leave
the cache with what SQLite held before a concurrent save (it now reads SQLite under the cache's
write lock), and two saves of a prompt's overrides could leave memory and SQLite with different
ones (the write lock is now held while SQLite is written).

## Not covered

- **Live monitoring.** Random judge audits, shadow promotion and drift demotion are covered by
  Rust unit tests and the real ONNX gateway HTTP test. The existing TLA+ lifecycle models abstract
  the student as deployed or absent; they do not model the audit window or its evidence policy.

- **Leases.** A worker that dies mid-job, or never gets the answer to its claim, leaves the job
  running until an admin cancels it (`check.sh --limits`), then retries it. The automatic retries
  don't help: only a failure the worker reports is retried, and a cancel only by an admin's Retry.
  The gateway would need a heartbeat and a way to fence off the old worker's late completion.
- **Crashes between SQLite writes.** A completion writes the job, the deployment and the status
  in three statements, a cancel in two; a gateway that dies in between restarts with them out of
  step. The models check consistency between requests, not after a crash.
- **Stopping a cancelled job's worker.** It trains to the end; its progress now gets a 409,
  which the worker could use to stop early.
- **Undeploy during training.** "Stop using student" doesn't cancel a running job, whose
  completion then deploys the new student.
- Several gateway replicas (the gateway runs as one instance), several prompts at once, the usage
  counters (`flush_usage` takes them and writes them in one transaction, with no `.await`
  between), and the System One calls and their escalation, which read the cached configuration
  and write nothing shared.
- Cache folders left partial by a gateway from before this change.
