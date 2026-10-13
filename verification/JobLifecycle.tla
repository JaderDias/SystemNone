---------------------------- MODULE JobLifecycle ----------------------------
(***************************************************************************)
(* The training jobs of one prompt: queue -> claim -> progress ->          *)
(* complete / fail / cancel, the retries of failed jobs, an admin's Retry  *)
(* of a cancelled job, and the student each job deploys.                   *)
(*                                                                         *)
(* Code: gateway/src/lib.rs (count_request, queue_job), gateway/src/api.rs *)
(* (train_now, undeploy, cancel_job, retry_job, fail_prompt,               *)
(* retry_or_fail, claim_job, job_progress, complete_job),                  *)
(* gateway/src/db.rs (claim_job, skip_wait, create_retry, finish_job,      *)
(* job_message) and src/system_none/worker.py (run_worker).                *)
(*                                                                         *)
(* The gateway runs on tokio's multi-threaded runtime, so handlers run in  *)
(* parallel: any two of them can interleave between their SQLite calls     *)
(* (each one locks the connection on its own) and their in-memory writes,  *)
(* not only at .await points. Each label below is one such step.           *)
(*                                                                         *)
(* Fixed = FALSE models the code before this change, Fixed = TRUE after.   *)
(* Retries (MaxAttempts > 1) and the admin's Retry came after the fixes:   *)
(* the model of the code before them has neither. Time isn't modelled: a   *)
(* claim may take a retry any time after it is queued, which includes      *)
(* every time after its not_before. claim_job's time check only removes    *)
(* behaviours, so every safety property holds with it too; "Train now"     *)
(* making a waiting retry claimable at once (db.skip_wait) changes nothing *)
(* the model tracks.                                                       *)
(***************************************************************************)
EXTENDS Naturals, FiniteSets, Sequences, TLC

CONSTANTS
    Workers,      \* `system-none worker` processes
    Handlers,     \* gateway tasks serving POST /internal/jobs/{id}/complete
    Admins,       \* admin sessions, one request at a time each
    Clients,      \* /v1/classify requests that reach count_request past the threshold
    MaxJobs,      \* jobs the model may create, retries included
    MaxAttempts,  \* settings.max_attempts: a failure the worker reports is retried until
                  \* this many attempts have failed in a row
    AdminOps,     \* requests per admin session
    Fixed,        \* TRUE: the code with this change's fixes; FALSE: before them
    WorkerCrash,  \* can a worker die mid-job (or lose the reply to its claim)?
    LostCalls     \* how many /complete calls may fail in transit

Jobs == 1..MaxJobs
NoJob == 0

(* --algorithm JobLifecycle

variables
    \* SQLite (db.rs)
    jobs = [j \in Jobs |-> "none"],   \* jobs.status: none (not created yet), queued, running, done, failed
    msg = [j \in Jobs |-> "none"],    \* jobs.message: none, progress, cancelled, failure (the worker's),
                                      \* unloadable (the student couldn't be loaded), result
    attempt = [j \in Jobs |-> 0],     \* jobs.attempt: 1, or one more than the failed job's for its retry
    retryOf = [j \in Jobs |-> NoJob], \* the failed job a retry follows (its spec's retry_of)
    adminRetryOf = [j \in Jobs |-> NoJob], \* the cancelled job an admin's Retry queued this one
                                      \* for (its spec's retry_of, with reason manual_retry)
    dbStatus = "collecting",          \* prompts.status
    depJob = NoJob,                   \* the prompt's row in `deployments`: job_id ...
    depActive = FALSE,                \* ... and active
    \* gateway memory (lib.rs)
    memStatus = "collecting",         \* PromptState.status
    memDep = NoJob,                   \* PromptState.deployed: the student /v1/classify uses
    lock = "free",                    \* AppState.queue_lock
    \* the store and the network
    uploaded = {},                    \* jobs whose student the worker put in the store
    inbox = {},                       \* /complete calls on their way to a handler: <<job, ok>>
    replied = {},                     \* jobs whose /complete call was answered
    lostCalls = 0,
    \* bookkeeping for the properties
    nextJob = 1,
    handedOut = [j \in Jobs |-> 0],   \* how many claims returned the job
    cancelled = {},                   \* jobs an admin's cancel marked failed
    spurious = FALSE;                 \* the threshold queued a job for a prompt that wasn't collecting

define
    Open == {j \in Jobs : jobs[j] \in {"queued", "running"}}
    Oldest(S) == CHOOSE j \in S : \A k \in S : j <= k
end define;

\* AppState::queue_job (lib.rs:298-311), under queue_lock. `threshold` is TRUE when
\* count_request (lib.rs:285-296) or requeue_ready (api.rs:587-606) calls it: both
\* checked that the prompt was collecting before taking the lock.
procedure QueueJob(threshold)
begin
QLock:
    await lock = "free";
    lock := self;
QOpen:      \* db.open_job_for (lib.rs:300); after the fix, the threshold path checks the status again
    if Open # {} \/ nextJob > MaxJobs \/ (Fixed /\ threshold /\ memStatus # "collecting") then
        lock := "free";
        return;
    end if;
QCreate:    \* db.create_job (lib.rs:303)
    spurious := spurious \/ (threshold /\ memStatus # "collecting");
    jobs[nextJob] := "queued";
    attempt[nextJob] := 1;
    nextJob := nextJob + 1;
QDbStatus:  \* db.set_status(queued) (lib.rs:307)
    dbStatus := "queued";
QMemStatus: \* p.set_status(queued) (lib.rs:308)
    memStatus := "queued";
    lock := "free";
    return;
end procedure;

\* fail_prompt (api.rs:786-797): a prompt that still has a student stays "deployed".
procedure FailPrompt()
variables st = "none";
begin
FRead:
    st := IF memDep # NoJob THEN "deployed" ELSE "failed";
FDb:
    dbStatus := st;
FMem:
    memStatus := st;
    return;
end procedure;

\* A /v1/classify request past the threshold: count_request (lib.rs:285-296).
\* requeue_ready, after a settings or override change, takes the same path.
\* Fairness: a request may never come (CCount:-), but once a handler starts, it finishes.
fair process client \in Clients
begin
CCount:-
    if memStatus = "collecting" then
        call QueueJob(TRUE);
    end if;
end process;

\* An admin session: "Train now", "Stop using student", cancel and, after the fixes, Retry of a
\* cancelled job, in any order. The admin may stop at any time (ANext:-); a request, once sent,
\* finishes.
fair process admin \in Admins
variables ops = 0, aj = NoJob;
begin
ANext:-
    while ops < AdminOps do
        ops := ops + 1;
        either
            call QueueJob(FALSE);       \* train_now (api.rs:732-748)
            goto ANext;
        or
            goto AUndeploy;
        or
            await \E k \in Jobs : jobs[k] # "none";
            goto ACancel;
        or
            await Fixed /\ cancelled # {};
            goto ARetry;
        end either;
    end while;
    goto ADone;
AUndeploy:  \* undeploy (api.rs:750-764)
    await ~Fixed \/ lock = "free";
    if Fixed then lock := self end if;
    memDep := NoJob;
AUndeployDb:
    depActive := FALSE;
AUndeployStatus:
    dbStatus := "collecting";
AUndeployMem:
    memStatus := "collecting";
    if Fixed then lock := "free" end if;
    goto ANext;
ACancel:    \* the admin picks a job from the list
    with j \in {k \in Jobs : jobs[k] # "none"} do
        aj := j;
    end with;
ACancelRead: \* cancel_job reads it (api.rs:772-780): 409 unless queued or running
    if jobs[aj] \notin {"queued", "running"} then
        goto ANext;
    end if;
ACancelFinish: \* db.finish_job(false, "cancelled by an admin") (api.rs:781); after the fix,
               \* under queue_lock and only if the job is still queued or running
    await ~Fixed \/ lock = "free";
    if Fixed /\ jobs[aj] \notin {"queued", "running"} then
        goto ANext;                     \* 409: it finished meanwhile
    else
        jobs[aj] := "failed";
        msg[aj] := "cancelled";
        cancelled := cancelled \cup {aj};
        if Fixed then lock := self end if;
    end if;
ACancelPrompt:
    call FailPrompt();
ACancelUnlock:
    if Fixed then lock := "free" end if;
    goto ANext;
ARetry:     \* retry_job: the admin's Retry of a job an admin cancelled (any other job gets a 409
            \* and changes nothing). Under queue_lock, nothing else changes a job's status, so
            \* reading the job and db.open_job_for are one step: 409 while the prompt has a job
            \* queued or running.
    await lock = "free";
    with j \in cancelled do
        aj := j;
    end with;
    if Open # {} \/ nextJob > MaxJobs then
        goto ANext;
    else
        lock := self;
    end if;
ARetryCreate: \* db.create_job: a new job, at attempt 1 and claimable at once, as "Train now"
              \* queues; the cancelled job stays as it is
    jobs[nextJob] := "queued";
    attempt[nextJob] := 1;
    adminRetryOf[nextJob] := aj;
    nextJob := nextJob + 1;
ARetryDb:   \* db.set_status(queued)
    dbStatus := "queued";
ARetryMem:  \* p.set_status(queued)
    memStatus := "queued";
    lock := "free";
    goto ANext;
ADone:
    skip;
end process;

\* A worker (worker.py:150-176) and the gateway handlers it calls synchronously.
fair process worker \in Workers
variables wj = NoJob, wok = FALSE;
begin
WClaim:     \* POST /internal/jobs/claim: db.claim_job (db.rs:485-513) is one transaction
    await (\E j \in Jobs : jobs[j] = "queued") /\ (~Fixed \/ lock = "free");
    with j = Oldest({k \in Jobs : jobs[k] = "queued"}) do
        wj := j;
        jobs[j] := "running";
        handedOut[j] := handedOut[j] + 1;
    end with;
    dbStatus := "training";
    if Fixed then lock := self end if;
WClaimMem:  \* p.set_status("training") (api.rs:805-807)
    memStatus := "training";
    if Fixed then lock := "free" end if;
WProgress:  \* process() posts progress (worker.py:128, 130): db.job_message (api.rs:825-834);
            \* after the fix, only while the job is running
    either
        await WorkerCrash;              \* the worker dies, or the claim's reply was lost
        goto WDead;
    or
        if ~Fixed \/ jobs[wj] = "running" then
            msg[wj] := "progress";
        end if;
    end either;
WTrain:     \* process() publishes the student to the store, or fails (worker.py:117-147)
    either
        uploaded := uploaded \cup {wj};
        wok := TRUE;
    or
        wok := FALSE;
    end either;
WSend:      \* POST .../complete (worker.py:174)
    either
        inbox := inbox \cup {<<wj, wok>>};
        goto WWait;
    or          \* connection refused: the gateway is restarting
        await lostCalls < LostCalls;
        lostCalls := lostCalls + 1;
        \* before the fix the httpx.TransportError ends run_worker; after it, the call is retried
        if Fixed then goto WSend else goto WDead end if;
    end either;
WWait:
    either
        await wj \in replied;
        goto WClaim;
    or          \* timeout or reset: the reply is lost, and the call may still be running
        await lostCalls < LostCalls;
        lostCalls := lostCalls + 1;
        if Fixed then goto WSend else goto WDead end if;
    end either;
WDead:
    skip;
end process;

\* complete_job (api.rs:845-900), one task per call.
fair process handler \in Handlers
variables hj = NoJob, hok = FALSE;
begin
HIdle:
    with r \in inbox do
        inbox := inbox \ {r};
        hj := r[1];
        hok := r[2];
    end with;
HCheck:     \* 409 unless the job is running (api.rs:852-857)
    if jobs[hj] # "running" then
        replied := replied \cup {hj};
        goto HIdle;
    elsif ~hok then
        goto HFail;
    end if;
HLoad:      \* load_student(...).await (api.rs:865): downloads the files, opens the ONNX session
    if hj \notin uploaded then
        goto HFail;                     \* 422
    elsif Fixed then
        goto HDone;
    end if;
    \* Before the fix (api.rs:886-894): nothing checks the job's status again after the await.
HPutDep:
    depJob := hj;
    depActive := TRUE;
HFinish:
    jobs[hj] := "done";
    msg[hj] := "result";
HDbStatus:
    dbStatus := "deployed";
HMemDep:
    memDep := hj;
HMemStatus:
    memStatus := "deployed";
    replied := replied \cup {hj};
    goto HIdle;
    \* After the fix: under queue_lock, finish_job moves only a running job, and the student
    \* is deployed only if it did.
HDone:
    await lock = "free";
    if jobs[hj] # "running" then
        replied := replied \cup {hj};
        goto HIdle;                     \* 409: cancelled, or completed by a repeated call
    else
        jobs[hj] := "done";
        msg[hj] := "result";
        lock := self;
    end if;
HDeploy:
    depJob := hj;
    depActive := TRUE;
HDeployStatus:
    dbStatus := "deployed";
HDeployMem:
    memDep := hj;
HDeployMemStatus:
    memStatus := "deployed";
    lock := "free";
    replied := replied \cup {hj};
    goto HIdle;
HFail:      \* db.finish_job(false) (api.rs:859-861, 869-871): the worker reported the failure
            \* (~hok), or the student it reported can't be loaded
    await ~Fixed \/ lock = "free";
    if Fixed /\ jobs[hj] # "running" then
        replied := replied \cup {hj};
        goto HIdle;
    else
        jobs[hj] := "failed";
        msg[hj] := IF hok THEN "unloadable" ELSE "failure";
        if Fixed then lock := self end if;
    end if;
HRetry:     \* retry_or_fail: while attempts are left, a failure the worker reports is retried:
            \* db.create_retry queues the next attempt (which waits for its time). The model
            \* has no job left for the retry of job MaxJobs, and fails the prompt instead.
    if Fixed /\ ~hok /\ attempt[hj] < MaxAttempts /\ nextJob <= MaxJobs then
        jobs[nextJob] := "queued";
        attempt[nextJob] := attempt[hj] + 1;
        retryOf[nextJob] := hj;
        nextJob := nextJob + 1;
    else
        goto HFailPrompt;
    end if;
HRetryDb:   \* db.set_status(queued)
    dbStatus := "queued";
HRetryMem:  \* p.set_status(queued)
    memStatus := "queued";
    lock := "free";
    replied := replied \cup {hj};
    goto HIdle;
HFailPrompt: \* fail_prompt: after the last attempt, a student that can't be loaded, or before the fixes
    call FailPrompt();
HFailUnlock:
    if Fixed then lock := "free" end if;
    replied := replied \cup {hj};
    goto HIdle;
end process;

end algorithm; *)
\* BEGIN TRANSLATION
CONSTANT defaultInitValue
VARIABLES jobs, msg, attempt, retryOf, adminRetryOf, dbStatus, depJob, 
          depActive, memStatus, memDep, lock, uploaded, inbox, replied, 
          lostCalls, nextJob, handedOut, cancelled, spurious, pc, stack

(* define statement *)
Open == {j \in Jobs : jobs[j] \in {"queued", "running"}}
Oldest(S) == CHOOSE j \in S : \A k \in S : j <= k

VARIABLES threshold, st, ops, aj, wj, wok, hj, hok

vars == << jobs, msg, attempt, retryOf, adminRetryOf, dbStatus, depJob, 
           depActive, memStatus, memDep, lock, uploaded, inbox, replied, 
           lostCalls, nextJob, handedOut, cancelled, spurious, pc, stack, 
           threshold, st, ops, aj, wj, wok, hj, hok >>

ProcSet == (Clients) \cup (Admins) \cup (Workers) \cup (Handlers)

Init == (* Global variables *)
        /\ jobs = [j \in Jobs |-> "none"]
        /\ msg = [j \in Jobs |-> "none"]
        /\ attempt = [j \in Jobs |-> 0]
        /\ retryOf = [j \in Jobs |-> NoJob]
        /\ adminRetryOf = [j \in Jobs |-> NoJob]
        /\ dbStatus = "collecting"
        /\ depJob = NoJob
        /\ depActive = FALSE
        /\ memStatus = "collecting"
        /\ memDep = NoJob
        /\ lock = "free"
        /\ uploaded = {}
        /\ inbox = {}
        /\ replied = {}
        /\ lostCalls = 0
        /\ nextJob = 1
        /\ handedOut = [j \in Jobs |-> 0]
        /\ cancelled = {}
        /\ spurious = FALSE
        (* Procedure QueueJob *)
        /\ threshold = [ self \in ProcSet |-> defaultInitValue]
        (* Procedure FailPrompt *)
        /\ st = [ self \in ProcSet |-> "none"]
        (* Process admin *)
        /\ ops = [self \in Admins |-> 0]
        /\ aj = [self \in Admins |-> NoJob]
        (* Process worker *)
        /\ wj = [self \in Workers |-> NoJob]
        /\ wok = [self \in Workers |-> FALSE]
        (* Process handler *)
        /\ hj = [self \in Handlers |-> NoJob]
        /\ hok = [self \in Handlers |-> FALSE]
        /\ stack = [self \in ProcSet |-> << >>]
        /\ pc = [self \in ProcSet |-> CASE self \in Clients -> "CCount"
                                        [] self \in Admins -> "ANext"
                                        [] self \in Workers -> "WClaim"
                                        [] self \in Handlers -> "HIdle"]

QLock(self) == /\ pc[self] = "QLock"
               /\ lock = "free"
               /\ lock' = self
               /\ pc' = [pc EXCEPT ![self] = "QOpen"]
               /\ UNCHANGED << jobs, msg, attempt, retryOf, adminRetryOf, 
                               dbStatus, depJob, depActive, memStatus, memDep, 
                               uploaded, inbox, replied, lostCalls, nextJob, 
                               handedOut, cancelled, spurious, stack, 
                               threshold, st, ops, aj, wj, wok, hj, hok >>

QOpen(self) == /\ pc[self] = "QOpen"
               /\ IF Open # {} \/ nextJob > MaxJobs \/ (Fixed /\ threshold[self] /\ memStatus # "collecting")
                     THEN /\ lock' = "free"
                          /\ pc' = [pc EXCEPT ![self] = Head(stack[self]).pc]
                          /\ threshold' = [threshold EXCEPT ![self] = Head(stack[self]).threshold]
                          /\ stack' = [stack EXCEPT ![self] = Tail(stack[self])]
                     ELSE /\ pc' = [pc EXCEPT ![self] = "QCreate"]
                          /\ UNCHANGED << lock, stack, threshold >>
               /\ UNCHANGED << jobs, msg, attempt, retryOf, adminRetryOf, 
                               dbStatus, depJob, depActive, memStatus, memDep, 
                               uploaded, inbox, replied, lostCalls, nextJob, 
                               handedOut, cancelled, spurious, st, ops, aj, wj, 
                               wok, hj, hok >>

QCreate(self) == /\ pc[self] = "QCreate"
                 /\ spurious' = (spurious \/ (threshold[self] /\ memStatus # "collecting"))
                 /\ jobs' = [jobs EXCEPT ![nextJob] = "queued"]
                 /\ attempt' = [attempt EXCEPT ![nextJob] = 1]
                 /\ nextJob' = nextJob + 1
                 /\ pc' = [pc EXCEPT ![self] = "QDbStatus"]
                 /\ UNCHANGED << msg, retryOf, adminRetryOf, dbStatus, depJob, 
                                 depActive, memStatus, memDep, lock, uploaded, 
                                 inbox, replied, lostCalls, handedOut, 
                                 cancelled, stack, threshold, st, ops, aj, wj, 
                                 wok, hj, hok >>

QDbStatus(self) == /\ pc[self] = "QDbStatus"
                   /\ dbStatus' = "queued"
                   /\ pc' = [pc EXCEPT ![self] = "QMemStatus"]
                   /\ UNCHANGED << jobs, msg, attempt, retryOf, adminRetryOf, 
                                   depJob, depActive, memStatus, memDep, lock, 
                                   uploaded, inbox, replied, lostCalls, 
                                   nextJob, handedOut, cancelled, spurious, 
                                   stack, threshold, st, ops, aj, wj, wok, hj, 
                                   hok >>

QMemStatus(self) == /\ pc[self] = "QMemStatus"
                    /\ memStatus' = "queued"
                    /\ lock' = "free"
                    /\ pc' = [pc EXCEPT ![self] = Head(stack[self]).pc]
                    /\ threshold' = [threshold EXCEPT ![self] = Head(stack[self]).threshold]
                    /\ stack' = [stack EXCEPT ![self] = Tail(stack[self])]
                    /\ UNCHANGED << jobs, msg, attempt, retryOf, adminRetryOf, 
                                    dbStatus, depJob, depActive, memDep, 
                                    uploaded, inbox, replied, lostCalls, 
                                    nextJob, handedOut, cancelled, spurious, 
                                    st, ops, aj, wj, wok, hj, hok >>

QueueJob(self) == QLock(self) \/ QOpen(self) \/ QCreate(self)
                     \/ QDbStatus(self) \/ QMemStatus(self)

FRead(self) == /\ pc[self] = "FRead"
               /\ st' = [st EXCEPT ![self] = IF memDep # NoJob THEN "deployed" ELSE "failed"]
               /\ pc' = [pc EXCEPT ![self] = "FDb"]
               /\ UNCHANGED << jobs, msg, attempt, retryOf, adminRetryOf, 
                               dbStatus, depJob, depActive, memStatus, memDep, 
                               lock, uploaded, inbox, replied, lostCalls, 
                               nextJob, handedOut, cancelled, spurious, stack, 
                               threshold, ops, aj, wj, wok, hj, hok >>

FDb(self) == /\ pc[self] = "FDb"
             /\ dbStatus' = st[self]
             /\ pc' = [pc EXCEPT ![self] = "FMem"]
             /\ UNCHANGED << jobs, msg, attempt, retryOf, adminRetryOf, depJob, 
                             depActive, memStatus, memDep, lock, uploaded, 
                             inbox, replied, lostCalls, nextJob, handedOut, 
                             cancelled, spurious, stack, threshold, st, ops, 
                             aj, wj, wok, hj, hok >>

FMem(self) == /\ pc[self] = "FMem"
              /\ memStatus' = st[self]
              /\ pc' = [pc EXCEPT ![self] = Head(stack[self]).pc]
              /\ st' = [st EXCEPT ![self] = Head(stack[self]).st]
              /\ stack' = [stack EXCEPT ![self] = Tail(stack[self])]
              /\ UNCHANGED << jobs, msg, attempt, retryOf, adminRetryOf, 
                              dbStatus, depJob, depActive, memDep, lock, 
                              uploaded, inbox, replied, lostCalls, nextJob, 
                              handedOut, cancelled, spurious, threshold, ops, 
                              aj, wj, wok, hj, hok >>

FailPrompt(self) == FRead(self) \/ FDb(self) \/ FMem(self)

CCount(self) == /\ pc[self] = "CCount"
                /\ IF memStatus = "collecting"
                      THEN /\ /\ stack' = [stack EXCEPT ![self] = << [ procedure |->  "QueueJob",
                                                                       pc        |->  "Done",
                                                                       threshold |->  threshold[self] ] >>
                                                                   \o stack[self]]
                              /\ threshold' = [threshold EXCEPT ![self] = TRUE]
                           /\ pc' = [pc EXCEPT ![self] = "QLock"]
                      ELSE /\ pc' = [pc EXCEPT ![self] = "Done"]
                           /\ UNCHANGED << stack, threshold >>
                /\ UNCHANGED << jobs, msg, attempt, retryOf, adminRetryOf, 
                                dbStatus, depJob, depActive, memStatus, memDep, 
                                lock, uploaded, inbox, replied, lostCalls, 
                                nextJob, handedOut, cancelled, spurious, st, 
                                ops, aj, wj, wok, hj, hok >>

client(self) == CCount(self)

ANext(self) == /\ pc[self] = "ANext"
               /\ IF ops[self] < AdminOps
                     THEN /\ ops' = [ops EXCEPT ![self] = ops[self] + 1]
                          /\ \/ /\ /\ stack' = [stack EXCEPT ![self] = << [ procedure |->  "QueueJob",
                                                                            pc        |->  "ANext",
                                                                            threshold |->  threshold[self] ] >>
                                                                        \o stack[self]]
                                   /\ threshold' = [threshold EXCEPT ![self] = FALSE]
                                /\ pc' = [pc EXCEPT ![self] = "QLock"]
                             \/ /\ pc' = [pc EXCEPT ![self] = "AUndeploy"]
                                /\ UNCHANGED <<stack, threshold>>
                             \/ /\ \E k \in Jobs : jobs[k] # "none"
                                /\ pc' = [pc EXCEPT ![self] = "ACancel"]
                                /\ UNCHANGED <<stack, threshold>>
                             \/ /\ Fixed /\ cancelled # {}
                                /\ pc' = [pc EXCEPT ![self] = "ARetry"]
                                /\ UNCHANGED <<stack, threshold>>
                     ELSE /\ pc' = [pc EXCEPT ![self] = "ADone"]
                          /\ UNCHANGED << stack, threshold, ops >>
               /\ UNCHANGED << jobs, msg, attempt, retryOf, adminRetryOf, 
                               dbStatus, depJob, depActive, memStatus, memDep, 
                               lock, uploaded, inbox, replied, lostCalls, 
                               nextJob, handedOut, cancelled, spurious, st, aj, 
                               wj, wok, hj, hok >>

AUndeploy(self) == /\ pc[self] = "AUndeploy"
                   /\ ~Fixed \/ lock = "free"
                   /\ IF Fixed
                         THEN /\ lock' = self
                         ELSE /\ TRUE
                              /\ lock' = lock
                   /\ memDep' = NoJob
                   /\ pc' = [pc EXCEPT ![self] = "AUndeployDb"]
                   /\ UNCHANGED << jobs, msg, attempt, retryOf, adminRetryOf, 
                                   dbStatus, depJob, depActive, memStatus, 
                                   uploaded, inbox, replied, lostCalls, 
                                   nextJob, handedOut, cancelled, spurious, 
                                   stack, threshold, st, ops, aj, wj, wok, hj, 
                                   hok >>

AUndeployDb(self) == /\ pc[self] = "AUndeployDb"
                     /\ depActive' = FALSE
                     /\ pc' = [pc EXCEPT ![self] = "AUndeployStatus"]
                     /\ UNCHANGED << jobs, msg, attempt, retryOf, adminRetryOf, 
                                     dbStatus, depJob, memStatus, memDep, lock, 
                                     uploaded, inbox, replied, lostCalls, 
                                     nextJob, handedOut, cancelled, spurious, 
                                     stack, threshold, st, ops, aj, wj, wok, 
                                     hj, hok >>

AUndeployStatus(self) == /\ pc[self] = "AUndeployStatus"
                         /\ dbStatus' = "collecting"
                         /\ pc' = [pc EXCEPT ![self] = "AUndeployMem"]
                         /\ UNCHANGED << jobs, msg, attempt, retryOf, 
                                         adminRetryOf, depJob, depActive, 
                                         memStatus, memDep, lock, uploaded, 
                                         inbox, replied, lostCalls, nextJob, 
                                         handedOut, cancelled, spurious, stack, 
                                         threshold, st, ops, aj, wj, wok, hj, 
                                         hok >>

AUndeployMem(self) == /\ pc[self] = "AUndeployMem"
                      /\ memStatus' = "collecting"
                      /\ IF Fixed
                            THEN /\ lock' = "free"
                            ELSE /\ TRUE
                                 /\ lock' = lock
                      /\ pc' = [pc EXCEPT ![self] = "ANext"]
                      /\ UNCHANGED << jobs, msg, attempt, retryOf, 
                                      adminRetryOf, dbStatus, depJob, 
                                      depActive, memDep, uploaded, inbox, 
                                      replied, lostCalls, nextJob, handedOut, 
                                      cancelled, spurious, stack, threshold, 
                                      st, ops, aj, wj, wok, hj, hok >>

ACancel(self) == /\ pc[self] = "ACancel"
                 /\ \E j \in {k \in Jobs : jobs[k] # "none"}:
                      aj' = [aj EXCEPT ![self] = j]
                 /\ pc' = [pc EXCEPT ![self] = "ACancelRead"]
                 /\ UNCHANGED << jobs, msg, attempt, retryOf, adminRetryOf, 
                                 dbStatus, depJob, depActive, memStatus, 
                                 memDep, lock, uploaded, inbox, replied, 
                                 lostCalls, nextJob, handedOut, cancelled, 
                                 spurious, stack, threshold, st, ops, wj, wok, 
                                 hj, hok >>

ACancelRead(self) == /\ pc[self] = "ACancelRead"
                     /\ IF jobs[aj[self]] \notin {"queued", "running"}
                           THEN /\ pc' = [pc EXCEPT ![self] = "ANext"]
                           ELSE /\ pc' = [pc EXCEPT ![self] = "ACancelFinish"]
                     /\ UNCHANGED << jobs, msg, attempt, retryOf, adminRetryOf, 
                                     dbStatus, depJob, depActive, memStatus, 
                                     memDep, lock, uploaded, inbox, replied, 
                                     lostCalls, nextJob, handedOut, cancelled, 
                                     spurious, stack, threshold, st, ops, aj, 
                                     wj, wok, hj, hok >>

ACancelFinish(self) == /\ pc[self] = "ACancelFinish"
                       /\ ~Fixed \/ lock = "free"
                       /\ IF Fixed /\ jobs[aj[self]] \notin {"queued", "running"}
                             THEN /\ pc' = [pc EXCEPT ![self] = "ANext"]
                                  /\ UNCHANGED << jobs, msg, lock, cancelled >>
                             ELSE /\ jobs' = [jobs EXCEPT ![aj[self]] = "failed"]
                                  /\ msg' = [msg EXCEPT ![aj[self]] = "cancelled"]
                                  /\ cancelled' = (cancelled \cup {aj[self]})
                                  /\ IF Fixed
                                        THEN /\ lock' = self
                                        ELSE /\ TRUE
                                             /\ lock' = lock
                                  /\ pc' = [pc EXCEPT ![self] = "ACancelPrompt"]
                       /\ UNCHANGED << attempt, retryOf, adminRetryOf, 
                                       dbStatus, depJob, depActive, memStatus, 
                                       memDep, uploaded, inbox, replied, 
                                       lostCalls, nextJob, handedOut, spurious, 
                                       stack, threshold, st, ops, aj, wj, wok, 
                                       hj, hok >>

ACancelPrompt(self) == /\ pc[self] = "ACancelPrompt"
                       /\ stack' = [stack EXCEPT ![self] = << [ procedure |->  "FailPrompt",
                                                                pc        |->  "ACancelUnlock",
                                                                st        |->  st[self] ] >>
                                                            \o stack[self]]
                       /\ st' = [st EXCEPT ![self] = "none"]
                       /\ pc' = [pc EXCEPT ![self] = "FRead"]
                       /\ UNCHANGED << jobs, msg, attempt, retryOf, 
                                       adminRetryOf, dbStatus, depJob, 
                                       depActive, memStatus, memDep, lock, 
                                       uploaded, inbox, replied, lostCalls, 
                                       nextJob, handedOut, cancelled, spurious, 
                                       threshold, ops, aj, wj, wok, hj, hok >>

ACancelUnlock(self) == /\ pc[self] = "ACancelUnlock"
                       /\ IF Fixed
                             THEN /\ lock' = "free"
                             ELSE /\ TRUE
                                  /\ lock' = lock
                       /\ pc' = [pc EXCEPT ![self] = "ANext"]
                       /\ UNCHANGED << jobs, msg, attempt, retryOf, 
                                       adminRetryOf, dbStatus, depJob, 
                                       depActive, memStatus, memDep, uploaded, 
                                       inbox, replied, lostCalls, nextJob, 
                                       handedOut, cancelled, spurious, stack, 
                                       threshold, st, ops, aj, wj, wok, hj, 
                                       hok >>

ARetry(self) == /\ pc[self] = "ARetry"
                /\ lock = "free"
                /\ \E j \in cancelled:
                     aj' = [aj EXCEPT ![self] = j]
                /\ IF Open # {} \/ nextJob > MaxJobs
                      THEN /\ pc' = [pc EXCEPT ![self] = "ANext"]
                           /\ lock' = lock
                      ELSE /\ lock' = self
                           /\ pc' = [pc EXCEPT ![self] = "ARetryCreate"]
                /\ UNCHANGED << jobs, msg, attempt, retryOf, adminRetryOf, 
                                dbStatus, depJob, depActive, memStatus, memDep, 
                                uploaded, inbox, replied, lostCalls, nextJob, 
                                handedOut, cancelled, spurious, stack, 
                                threshold, st, ops, wj, wok, hj, hok >>

ARetryCreate(self) == /\ pc[self] = "ARetryCreate"
                      /\ jobs' = [jobs EXCEPT ![nextJob] = "queued"]
                      /\ attempt' = [attempt EXCEPT ![nextJob] = 1]
                      /\ adminRetryOf' = [adminRetryOf EXCEPT ![nextJob] = aj[self]]
                      /\ nextJob' = nextJob + 1
                      /\ pc' = [pc EXCEPT ![self] = "ARetryDb"]
                      /\ UNCHANGED << msg, retryOf, dbStatus, depJob, 
                                      depActive, memStatus, memDep, lock, 
                                      uploaded, inbox, replied, lostCalls, 
                                      handedOut, cancelled, spurious, stack, 
                                      threshold, st, ops, aj, wj, wok, hj, hok >>

ARetryDb(self) == /\ pc[self] = "ARetryDb"
                  /\ dbStatus' = "queued"
                  /\ pc' = [pc EXCEPT ![self] = "ARetryMem"]
                  /\ UNCHANGED << jobs, msg, attempt, retryOf, adminRetryOf, 
                                  depJob, depActive, memStatus, memDep, lock, 
                                  uploaded, inbox, replied, lostCalls, nextJob, 
                                  handedOut, cancelled, spurious, stack, 
                                  threshold, st, ops, aj, wj, wok, hj, hok >>

ARetryMem(self) == /\ pc[self] = "ARetryMem"
                   /\ memStatus' = "queued"
                   /\ lock' = "free"
                   /\ pc' = [pc EXCEPT ![self] = "ANext"]
                   /\ UNCHANGED << jobs, msg, attempt, retryOf, adminRetryOf, 
                                   dbStatus, depJob, depActive, memDep, 
                                   uploaded, inbox, replied, lostCalls, 
                                   nextJob, handedOut, cancelled, spurious, 
                                   stack, threshold, st, ops, aj, wj, wok, hj, 
                                   hok >>

ADone(self) == /\ pc[self] = "ADone"
               /\ TRUE
               /\ pc' = [pc EXCEPT ![self] = "Done"]
               /\ UNCHANGED << jobs, msg, attempt, retryOf, adminRetryOf, 
                               dbStatus, depJob, depActive, memStatus, memDep, 
                               lock, uploaded, inbox, replied, lostCalls, 
                               nextJob, handedOut, cancelled, spurious, stack, 
                               threshold, st, ops, aj, wj, wok, hj, hok >>

admin(self) == ANext(self) \/ AUndeploy(self) \/ AUndeployDb(self)
                  \/ AUndeployStatus(self) \/ AUndeployMem(self)
                  \/ ACancel(self) \/ ACancelRead(self)
                  \/ ACancelFinish(self) \/ ACancelPrompt(self)
                  \/ ACancelUnlock(self) \/ ARetry(self)
                  \/ ARetryCreate(self) \/ ARetryDb(self)
                  \/ ARetryMem(self) \/ ADone(self)

WClaim(self) == /\ pc[self] = "WClaim"
                /\ (\E j \in Jobs : jobs[j] = "queued") /\ (~Fixed \/ lock = "free")
                /\ LET j == Oldest({k \in Jobs : jobs[k] = "queued"}) IN
                     /\ wj' = [wj EXCEPT ![self] = j]
                     /\ jobs' = [jobs EXCEPT ![j] = "running"]
                     /\ handedOut' = [handedOut EXCEPT ![j] = handedOut[j] + 1]
                /\ dbStatus' = "training"
                /\ IF Fixed
                      THEN /\ lock' = self
                      ELSE /\ TRUE
                           /\ lock' = lock
                /\ pc' = [pc EXCEPT ![self] = "WClaimMem"]
                /\ UNCHANGED << msg, attempt, retryOf, adminRetryOf, depJob, 
                                depActive, memStatus, memDep, uploaded, inbox, 
                                replied, lostCalls, nextJob, cancelled, 
                                spurious, stack, threshold, st, ops, aj, wok, 
                                hj, hok >>

WClaimMem(self) == /\ pc[self] = "WClaimMem"
                   /\ memStatus' = "training"
                   /\ IF Fixed
                         THEN /\ lock' = "free"
                         ELSE /\ TRUE
                              /\ lock' = lock
                   /\ pc' = [pc EXCEPT ![self] = "WProgress"]
                   /\ UNCHANGED << jobs, msg, attempt, retryOf, adminRetryOf, 
                                   dbStatus, depJob, depActive, memDep, 
                                   uploaded, inbox, replied, lostCalls, 
                                   nextJob, handedOut, cancelled, spurious, 
                                   stack, threshold, st, ops, aj, wj, wok, hj, 
                                   hok >>

WProgress(self) == /\ pc[self] = "WProgress"
                   /\ \/ /\ WorkerCrash
                         /\ pc' = [pc EXCEPT ![self] = "WDead"]
                         /\ msg' = msg
                      \/ /\ IF ~Fixed \/ jobs[wj[self]] = "running"
                               THEN /\ msg' = [msg EXCEPT ![wj[self]] = "progress"]
                               ELSE /\ TRUE
                                    /\ msg' = msg
                         /\ pc' = [pc EXCEPT ![self] = "WTrain"]
                   /\ UNCHANGED << jobs, attempt, retryOf, adminRetryOf, 
                                   dbStatus, depJob, depActive, memStatus, 
                                   memDep, lock, uploaded, inbox, replied, 
                                   lostCalls, nextJob, handedOut, cancelled, 
                                   spurious, stack, threshold, st, ops, aj, wj, 
                                   wok, hj, hok >>

WTrain(self) == /\ pc[self] = "WTrain"
                /\ \/ /\ uploaded' = (uploaded \cup {wj[self]})
                      /\ wok' = [wok EXCEPT ![self] = TRUE]
                   \/ /\ wok' = [wok EXCEPT ![self] = FALSE]
                      /\ UNCHANGED uploaded
                /\ pc' = [pc EXCEPT ![self] = "WSend"]
                /\ UNCHANGED << jobs, msg, attempt, retryOf, adminRetryOf, 
                                dbStatus, depJob, depActive, memStatus, memDep, 
                                lock, inbox, replied, lostCalls, nextJob, 
                                handedOut, cancelled, spurious, stack, 
                                threshold, st, ops, aj, wj, hj, hok >>

WSend(self) == /\ pc[self] = "WSend"
               /\ \/ /\ inbox' = (inbox \cup {<<wj[self], wok[self]>>})
                     /\ pc' = [pc EXCEPT ![self] = "WWait"]
                     /\ UNCHANGED lostCalls
                  \/ /\ lostCalls < LostCalls
                     /\ lostCalls' = lostCalls + 1
                     /\ IF Fixed
                           THEN /\ pc' = [pc EXCEPT ![self] = "WSend"]
                           ELSE /\ pc' = [pc EXCEPT ![self] = "WDead"]
                     /\ inbox' = inbox
               /\ UNCHANGED << jobs, msg, attempt, retryOf, adminRetryOf, 
                               dbStatus, depJob, depActive, memStatus, memDep, 
                               lock, uploaded, replied, nextJob, handedOut, 
                               cancelled, spurious, stack, threshold, st, ops, 
                               aj, wj, wok, hj, hok >>

WWait(self) == /\ pc[self] = "WWait"
               /\ \/ /\ wj[self] \in replied
                     /\ pc' = [pc EXCEPT ![self] = "WClaim"]
                     /\ UNCHANGED lostCalls
                  \/ /\ lostCalls < LostCalls
                     /\ lostCalls' = lostCalls + 1
                     /\ IF Fixed
                           THEN /\ pc' = [pc EXCEPT ![self] = "WSend"]
                           ELSE /\ pc' = [pc EXCEPT ![self] = "WDead"]
               /\ UNCHANGED << jobs, msg, attempt, retryOf, adminRetryOf, 
                               dbStatus, depJob, depActive, memStatus, memDep, 
                               lock, uploaded, inbox, replied, nextJob, 
                               handedOut, cancelled, spurious, stack, 
                               threshold, st, ops, aj, wj, wok, hj, hok >>

WDead(self) == /\ pc[self] = "WDead"
               /\ TRUE
               /\ pc' = [pc EXCEPT ![self] = "Done"]
               /\ UNCHANGED << jobs, msg, attempt, retryOf, adminRetryOf, 
                               dbStatus, depJob, depActive, memStatus, memDep, 
                               lock, uploaded, inbox, replied, lostCalls, 
                               nextJob, handedOut, cancelled, spurious, stack, 
                               threshold, st, ops, aj, wj, wok, hj, hok >>

worker(self) == WClaim(self) \/ WClaimMem(self) \/ WProgress(self)
                   \/ WTrain(self) \/ WSend(self) \/ WWait(self)
                   \/ WDead(self)

HIdle(self) == /\ pc[self] = "HIdle"
               /\ \E r \in inbox:
                    /\ inbox' = inbox \ {r}
                    /\ hj' = [hj EXCEPT ![self] = r[1]]
                    /\ hok' = [hok EXCEPT ![self] = r[2]]
               /\ pc' = [pc EXCEPT ![self] = "HCheck"]
               /\ UNCHANGED << jobs, msg, attempt, retryOf, adminRetryOf, 
                               dbStatus, depJob, depActive, memStatus, memDep, 
                               lock, uploaded, replied, lostCalls, nextJob, 
                               handedOut, cancelled, spurious, stack, 
                               threshold, st, ops, aj, wj, wok >>

HCheck(self) == /\ pc[self] = "HCheck"
                /\ IF jobs[hj[self]] # "running"
                      THEN /\ replied' = (replied \cup {hj[self]})
                           /\ pc' = [pc EXCEPT ![self] = "HIdle"]
                      ELSE /\ IF ~hok[self]
                                 THEN /\ pc' = [pc EXCEPT ![self] = "HFail"]
                                 ELSE /\ pc' = [pc EXCEPT ![self] = "HLoad"]
                           /\ UNCHANGED replied
                /\ UNCHANGED << jobs, msg, attempt, retryOf, adminRetryOf, 
                                dbStatus, depJob, depActive, memStatus, memDep, 
                                lock, uploaded, inbox, lostCalls, nextJob, 
                                handedOut, cancelled, spurious, stack, 
                                threshold, st, ops, aj, wj, wok, hj, hok >>

HLoad(self) == /\ pc[self] = "HLoad"
               /\ IF hj[self] \notin uploaded
                     THEN /\ pc' = [pc EXCEPT ![self] = "HFail"]
                     ELSE /\ IF Fixed
                                THEN /\ pc' = [pc EXCEPT ![self] = "HDone"]
                                ELSE /\ pc' = [pc EXCEPT ![self] = "HPutDep"]
               /\ UNCHANGED << jobs, msg, attempt, retryOf, adminRetryOf, 
                               dbStatus, depJob, depActive, memStatus, memDep, 
                               lock, uploaded, inbox, replied, lostCalls, 
                               nextJob, handedOut, cancelled, spurious, stack, 
                               threshold, st, ops, aj, wj, wok, hj, hok >>

HPutDep(self) == /\ pc[self] = "HPutDep"
                 /\ depJob' = hj[self]
                 /\ depActive' = TRUE
                 /\ pc' = [pc EXCEPT ![self] = "HFinish"]
                 /\ UNCHANGED << jobs, msg, attempt, retryOf, adminRetryOf, 
                                 dbStatus, memStatus, memDep, lock, uploaded, 
                                 inbox, replied, lostCalls, nextJob, handedOut, 
                                 cancelled, spurious, stack, threshold, st, 
                                 ops, aj, wj, wok, hj, hok >>

HFinish(self) == /\ pc[self] = "HFinish"
                 /\ jobs' = [jobs EXCEPT ![hj[self]] = "done"]
                 /\ msg' = [msg EXCEPT ![hj[self]] = "result"]
                 /\ pc' = [pc EXCEPT ![self] = "HDbStatus"]
                 /\ UNCHANGED << attempt, retryOf, adminRetryOf, dbStatus, 
                                 depJob, depActive, memStatus, memDep, lock, 
                                 uploaded, inbox, replied, lostCalls, nextJob, 
                                 handedOut, cancelled, spurious, stack, 
                                 threshold, st, ops, aj, wj, wok, hj, hok >>

HDbStatus(self) == /\ pc[self] = "HDbStatus"
                   /\ dbStatus' = "deployed"
                   /\ pc' = [pc EXCEPT ![self] = "HMemDep"]
                   /\ UNCHANGED << jobs, msg, attempt, retryOf, adminRetryOf, 
                                   depJob, depActive, memStatus, memDep, lock, 
                                   uploaded, inbox, replied, lostCalls, 
                                   nextJob, handedOut, cancelled, spurious, 
                                   stack, threshold, st, ops, aj, wj, wok, hj, 
                                   hok >>

HMemDep(self) == /\ pc[self] = "HMemDep"
                 /\ memDep' = hj[self]
                 /\ pc' = [pc EXCEPT ![self] = "HMemStatus"]
                 /\ UNCHANGED << jobs, msg, attempt, retryOf, adminRetryOf, 
                                 dbStatus, depJob, depActive, memStatus, lock, 
                                 uploaded, inbox, replied, lostCalls, nextJob, 
                                 handedOut, cancelled, spurious, stack, 
                                 threshold, st, ops, aj, wj, wok, hj, hok >>

HMemStatus(self) == /\ pc[self] = "HMemStatus"
                    /\ memStatus' = "deployed"
                    /\ replied' = (replied \cup {hj[self]})
                    /\ pc' = [pc EXCEPT ![self] = "HIdle"]
                    /\ UNCHANGED << jobs, msg, attempt, retryOf, adminRetryOf, 
                                    dbStatus, depJob, depActive, memDep, lock, 
                                    uploaded, inbox, lostCalls, nextJob, 
                                    handedOut, cancelled, spurious, stack, 
                                    threshold, st, ops, aj, wj, wok, hj, hok >>

HDone(self) == /\ pc[self] = "HDone"
               /\ lock = "free"
               /\ IF jobs[hj[self]] # "running"
                     THEN /\ replied' = (replied \cup {hj[self]})
                          /\ pc' = [pc EXCEPT ![self] = "HIdle"]
                          /\ UNCHANGED << jobs, msg, lock >>
                     ELSE /\ jobs' = [jobs EXCEPT ![hj[self]] = "done"]
                          /\ msg' = [msg EXCEPT ![hj[self]] = "result"]
                          /\ lock' = self
                          /\ pc' = [pc EXCEPT ![self] = "HDeploy"]
                          /\ UNCHANGED replied
               /\ UNCHANGED << attempt, retryOf, adminRetryOf, dbStatus, 
                               depJob, depActive, memStatus, memDep, uploaded, 
                               inbox, lostCalls, nextJob, handedOut, cancelled, 
                               spurious, stack, threshold, st, ops, aj, wj, 
                               wok, hj, hok >>

HDeploy(self) == /\ pc[self] = "HDeploy"
                 /\ depJob' = hj[self]
                 /\ depActive' = TRUE
                 /\ pc' = [pc EXCEPT ![self] = "HDeployStatus"]
                 /\ UNCHANGED << jobs, msg, attempt, retryOf, adminRetryOf, 
                                 dbStatus, memStatus, memDep, lock, uploaded, 
                                 inbox, replied, lostCalls, nextJob, handedOut, 
                                 cancelled, spurious, stack, threshold, st, 
                                 ops, aj, wj, wok, hj, hok >>

HDeployStatus(self) == /\ pc[self] = "HDeployStatus"
                       /\ dbStatus' = "deployed"
                       /\ pc' = [pc EXCEPT ![self] = "HDeployMem"]
                       /\ UNCHANGED << jobs, msg, attempt, retryOf, 
                                       adminRetryOf, depJob, depActive, 
                                       memStatus, memDep, lock, uploaded, 
                                       inbox, replied, lostCalls, nextJob, 
                                       handedOut, cancelled, spurious, stack, 
                                       threshold, st, ops, aj, wj, wok, hj, 
                                       hok >>

HDeployMem(self) == /\ pc[self] = "HDeployMem"
                    /\ memDep' = hj[self]
                    /\ pc' = [pc EXCEPT ![self] = "HDeployMemStatus"]
                    /\ UNCHANGED << jobs, msg, attempt, retryOf, adminRetryOf, 
                                    dbStatus, depJob, depActive, memStatus, 
                                    lock, uploaded, inbox, replied, lostCalls, 
                                    nextJob, handedOut, cancelled, spurious, 
                                    stack, threshold, st, ops, aj, wj, wok, hj, 
                                    hok >>

HDeployMemStatus(self) == /\ pc[self] = "HDeployMemStatus"
                          /\ memStatus' = "deployed"
                          /\ lock' = "free"
                          /\ replied' = (replied \cup {hj[self]})
                          /\ pc' = [pc EXCEPT ![self] = "HIdle"]
                          /\ UNCHANGED << jobs, msg, attempt, retryOf, 
                                          adminRetryOf, dbStatus, depJob, 
                                          depActive, memDep, uploaded, inbox, 
                                          lostCalls, nextJob, handedOut, 
                                          cancelled, spurious, stack, 
                                          threshold, st, ops, aj, wj, wok, hj, 
                                          hok >>

HFail(self) == /\ pc[self] = "HFail"
               /\ ~Fixed \/ lock = "free"
               /\ IF Fixed /\ jobs[hj[self]] # "running"
                     THEN /\ replied' = (replied \cup {hj[self]})
                          /\ pc' = [pc EXCEPT ![self] = "HIdle"]
                          /\ UNCHANGED << jobs, msg, lock >>
                     ELSE /\ jobs' = [jobs EXCEPT ![hj[self]] = "failed"]
                          /\ msg' = [msg EXCEPT ![hj[self]] = IF hok[self] THEN "unloadable" ELSE "failure"]
                          /\ IF Fixed
                                THEN /\ lock' = self
                                ELSE /\ TRUE
                                     /\ lock' = lock
                          /\ pc' = [pc EXCEPT ![self] = "HRetry"]
                          /\ UNCHANGED replied
               /\ UNCHANGED << attempt, retryOf, adminRetryOf, dbStatus, 
                               depJob, depActive, memStatus, memDep, uploaded, 
                               inbox, lostCalls, nextJob, handedOut, cancelled, 
                               spurious, stack, threshold, st, ops, aj, wj, 
                               wok, hj, hok >>

HRetry(self) == /\ pc[self] = "HRetry"
                /\ IF Fixed /\ ~hok[self] /\ attempt[hj[self]] < MaxAttempts /\ nextJob <= MaxJobs
                      THEN /\ jobs' = [jobs EXCEPT ![nextJob] = "queued"]
                           /\ attempt' = [attempt EXCEPT ![nextJob] = attempt[hj[self]] + 1]
                           /\ retryOf' = [retryOf EXCEPT ![nextJob] = hj[self]]
                           /\ nextJob' = nextJob + 1
                           /\ pc' = [pc EXCEPT ![self] = "HRetryDb"]
                      ELSE /\ pc' = [pc EXCEPT ![self] = "HFailPrompt"]
                           /\ UNCHANGED << jobs, attempt, retryOf, nextJob >>
                /\ UNCHANGED << msg, adminRetryOf, dbStatus, depJob, depActive, 
                                memStatus, memDep, lock, uploaded, inbox, 
                                replied, lostCalls, handedOut, cancelled, 
                                spurious, stack, threshold, st, ops, aj, wj, 
                                wok, hj, hok >>

HRetryDb(self) == /\ pc[self] = "HRetryDb"
                  /\ dbStatus' = "queued"
                  /\ pc' = [pc EXCEPT ![self] = "HRetryMem"]
                  /\ UNCHANGED << jobs, msg, attempt, retryOf, adminRetryOf, 
                                  depJob, depActive, memStatus, memDep, lock, 
                                  uploaded, inbox, replied, lostCalls, nextJob, 
                                  handedOut, cancelled, spurious, stack, 
                                  threshold, st, ops, aj, wj, wok, hj, hok >>

HRetryMem(self) == /\ pc[self] = "HRetryMem"
                   /\ memStatus' = "queued"
                   /\ lock' = "free"
                   /\ replied' = (replied \cup {hj[self]})
                   /\ pc' = [pc EXCEPT ![self] = "HIdle"]
                   /\ UNCHANGED << jobs, msg, attempt, retryOf, adminRetryOf, 
                                   dbStatus, depJob, depActive, memDep, 
                                   uploaded, inbox, lostCalls, nextJob, 
                                   handedOut, cancelled, spurious, stack, 
                                   threshold, st, ops, aj, wj, wok, hj, hok >>

HFailPrompt(self) == /\ pc[self] = "HFailPrompt"
                     /\ stack' = [stack EXCEPT ![self] = << [ procedure |->  "FailPrompt",
                                                              pc        |->  "HFailUnlock",
                                                              st        |->  st[self] ] >>
                                                          \o stack[self]]
                     /\ st' = [st EXCEPT ![self] = "none"]
                     /\ pc' = [pc EXCEPT ![self] = "FRead"]
                     /\ UNCHANGED << jobs, msg, attempt, retryOf, adminRetryOf, 
                                     dbStatus, depJob, depActive, memStatus, 
                                     memDep, lock, uploaded, inbox, replied, 
                                     lostCalls, nextJob, handedOut, cancelled, 
                                     spurious, threshold, ops, aj, wj, wok, hj, 
                                     hok >>

HFailUnlock(self) == /\ pc[self] = "HFailUnlock"
                     /\ IF Fixed
                           THEN /\ lock' = "free"
                           ELSE /\ TRUE
                                /\ lock' = lock
                     /\ replied' = (replied \cup {hj[self]})
                     /\ pc' = [pc EXCEPT ![self] = "HIdle"]
                     /\ UNCHANGED << jobs, msg, attempt, retryOf, adminRetryOf, 
                                     dbStatus, depJob, depActive, memStatus, 
                                     memDep, uploaded, inbox, lostCalls, 
                                     nextJob, handedOut, cancelled, spurious, 
                                     stack, threshold, st, ops, aj, wj, wok, 
                                     hj, hok >>

handler(self) == HIdle(self) \/ HCheck(self) \/ HLoad(self)
                    \/ HPutDep(self) \/ HFinish(self) \/ HDbStatus(self)
                    \/ HMemDep(self) \/ HMemStatus(self) \/ HDone(self)
                    \/ HDeploy(self) \/ HDeployStatus(self)
                    \/ HDeployMem(self) \/ HDeployMemStatus(self)
                    \/ HFail(self) \/ HRetry(self) \/ HRetryDb(self)
                    \/ HRetryMem(self) \/ HFailPrompt(self)
                    \/ HFailUnlock(self)

(* Allow infinite stuttering to prevent deadlock on termination. *)
Terminating == /\ \A self \in ProcSet: pc[self] = "Done"
               /\ UNCHANGED vars

Next == (\E self \in ProcSet: QueueJob(self) \/ FailPrompt(self))
           \/ (\E self \in Clients: client(self))
           \/ (\E self \in Admins: admin(self))
           \/ (\E self \in Workers: worker(self))
           \/ (\E self \in Handlers: handler(self))
           \/ Terminating

Spec == /\ Init /\ [][Next]_vars
        /\ \A self \in Clients : /\ WF_vars((pc[self] # "CCount") /\ client(self))
                                 /\ WF_vars(QueueJob(self))
        /\ \A self \in Admins : /\ WF_vars((pc[self] # "ANext") /\ admin(self))
                                /\ WF_vars(QueueJob(self))
                                /\ WF_vars(FailPrompt(self))
        /\ \A self \in Workers : WF_vars(worker(self))
        /\ \A self \in Handlers : WF_vars(handler(self)) /\ WF_vars(FailPrompt(self))

Termination == <>(\A self \in ProcSet: pc[self] = "Done")

\* END TRANSLATION

-----------------------------------------------------------------------------
(* Properties *)

Statuses == {"collecting", "queued", "training", "deployed", "failed"}

TypeOK ==
    /\ jobs \in [Jobs -> {"none", "queued", "running", "done", "failed"}]
    /\ msg \in [Jobs -> {"none", "progress", "cancelled", "failure", "unloadable", "result"}]
    /\ attempt \in [Jobs -> Nat] /\ retryOf \in [Jobs -> Jobs \cup {NoJob}]
    /\ adminRetryOf \in [Jobs -> Jobs \cup {NoJob}]
    /\ nextJob \in 1..(Cardinality(Jobs) + 1)
    /\ dbStatus \in Statuses /\ memStatus \in Statuses
    /\ depJob \in Jobs \cup {NoJob} /\ depActive \in BOOLEAN
    /\ memDep \in Jobs \cup {NoJob}
    /\ lock \in {"free"} \cup ProcSet

\* At most one queued or running job per prompt.
AtMostOneOpenJob == Cardinality(Open) <= 1

\* No job is handed to two workers.
ClaimedOnce == \A j \in Jobs : handedOut[j] <= 1

\* The student /v1/classify uses comes from a job that completed successfully.
ServedIsDone == memDep # NoJob => jobs[memDep] = "done"

\* A job an admin cancelled never deploys its student.
CancelledNeverDeploys ==
    \A j \in cancelled : memDep # j /\ ~(depActive /\ depJob = j)

\* The request threshold queues a job only for a prompt that is collecting.
ThresholdOnlyWhileCollecting == ~spurious

\* An automatic retry follows one failure the worker reported (not a cancel, nor a student that
\* couldn't be loaded), as its next attempt.
RetriesFollowWorkerFailures ==
    \A j \in Jobs : retryOf[j] # NoJob =>
        /\ jobs[retryOf[j]] = "failed" /\ msg[retryOf[j]] = "failure"
        /\ attempt[j] = attempt[retryOf[j]] + 1
        /\ \A k \in Jobs : retryOf[k] = retryOf[j] => k = j

\* A job an admin cancelled is never retried automatically: only an admin's Retry queues its
\* prompt again (AdminRetriesFollowCancels).
CancelledNeverRetried == \A j \in Jobs : retryOf[j] \notin cancelled

\* A job an admin's Retry queued follows a job an admin cancelled, which stays cancelled, and
\* starts over at attempt 1.
AdminRetriesFollowCancels ==
    \A j \in Jobs : adminRetryOf[j] # NoJob =>
        /\ adminRetryOf[j] \in cancelled
        /\ jobs[adminRetryOf[j]] = "failed" /\ msg[adminRetryOf[j]] = "cancelled"
        /\ attempt[j] = 1 /\ retryOf[j] = NoJob

\* No training gets more than MaxAttempts attempts.
AttemptsBounded == \A j \in Jobs : attempt[j] <= MaxAttempts

\* No handler is between two of its steps.
Quiescent ==
    \A p \in ProcSet :
        pc[p] \in {"CCount", "ANext", "ADone", "WClaim", "WProgress", "WTrain",
                   "WSend", "WWait", "WDead", "HIdle", "Done"}

\* Between requests, every failure the worker reported before the last attempt has its retry:
\* a prompt fails only after its last attempt, a cancel, or a student that couldn't be loaded.
\* (Job MaxJobs has no job left for its retry. A job below it has: no job is created while it
\* is open, so its retry would be the next.)
RetriedUntilTheLastAttempt ==
    Quiescent =>
        \A j \in Jobs : (jobs[j] = "failed" /\ msg[j] = "failure" /\ attempt[j] < MaxAttempts
                         /\ j < MaxJobs)
            => \E k \in Jobs : retryOf[k] = j

\* Between requests, memory says what SQLite says.
MemoryMatchesDb ==
    Quiescent =>
        /\ memStatus = dbStatus
        /\ memDep = (IF depActive THEN depJob ELSE NoJob)

\* Between requests, a "queued" or "training" prompt has a job in that state.
StatusHasJob ==
    Quiescent =>
        /\ dbStatus = "queued" => \E j \in Jobs : jobs[j] = "queued"
        /\ dbStatus = "training" => \E j \in Jobs : jobs[j] = "running"

\* A finished job's status and message never change.
FinishedJobsStay ==
    [][\A j \in Jobs : jobs[j] \in {"done", "failed"} =>
            jobs'[j] = jobs[j] /\ msg'[j] = msg[j]]_vars

\* Every claimed job eventually finishes.
ClaimedJobsFinish == \A j \in Jobs : jobs[j] = "running" ~> jobs[j] \in {"done", "failed"}

\* Every queued job is eventually claimed or cancelled.
QueuedJobsLeaveTheQueue == \A j \in Jobs : jobs[j] = "queued" ~> jobs[j] # "queued"

\* queue_lock is always released.
LockReleased == lock # "free" ~> lock = "free"

=============================================================================
