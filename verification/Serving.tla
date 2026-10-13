------------------------------- MODULE Serving -------------------------------
(***************************************************************************)
(* Serving a prompt's student while it changes.                           *)
(*                                                                         *)
(* 1. Hot swap: /v1/classify clones PromptState.deployed (an               *)
(*    Arc<Deployed>: the student and its threshold) and predicts on its    *)
(*    clone, while complete_job and undeploy replace it (lib.rs:72-77,     *)
(*    api.rs:173-197, 759, 889-893).                                       *)
(* 2. The student download: load_student downloads a job's files into      *)
(*    cache/{prompt}/{job}/ unless model.onnx is there already, then opens *)
(*    them (lib.rs:349-363, store.rs:119-128). Two loads of the same job   *)
(*    happen when /complete is called again for it, or at a restart, which *)
(*    loads every active deployment.                                       *)
(*                                                                         *)
(* Fixed = FALSE models the code before this change, Fixed = TRUE after:   *)
(* the files are downloaded into a temporary folder that is renamed into   *)
(* place once complete.                                                    *)
(***************************************************************************)
EXTENDS Naturals, FiniteSets, Sequences, TLC

CONSTANTS
    Students,     \* students the prompt can be given, one per completed job
    Requests,     \* /v1/classify requests
    Deployer,     \* the task that sets PromptState.deployed: complete_job or undeploy
    Swaps,        \* how many times the deployer acts
    Loaders,      \* load_student calls for the same job
    Fixed,        \* TRUE: the code with this change's fixes; FALSE: before them
    StoreErrors   \* how many store reads may fail

NoStudent == "none"
Files == <<"model.onnx", "tokenizer.json", "manifest.json">>   \* MODEL_FILES, in download order
FileSet == {Files[k] : k \in 1..Len(Files)}

(* --algorithm Serving

variables
    \* 1. hot swap
    slot = NoStudent,                    \* PromptState.deployed
    refs = [s \in Students |-> 0],       \* the Arc's strong count
    alive = [s \in Students |-> FALSE],  \* the Deployed (student, ONNX session) is not dropped
    used = {},                           \* students deployed so far (each job deploys once)
    \* 2. the download
    complete = {},                       \* files in the job's cache folder whose last write finished
    writers = [f \in FileSet |-> 0],     \* writes in progress: the file exists but is truncated
    dirReady = FALSE,                    \* after the fix: the folder was renamed into place
    storeErrors = 0,
    result = [l \in Loaders |-> "none"], \* none, loaded, failed
    ownError = [l \in Loaders |-> FALSE];\* this load's own store read failed

define
    Drop(rf, s) == IF s = NoStudent THEN rf ELSE [rf EXCEPT ![s] = @ - 1]
    \* load_student skips the download (lib.rs:357)
    InCache == IF Fixed THEN dirReady
               ELSE "model.onnx" \in complete \/ writers["model.onnx"] > 0
    \* Student::load finds all three files, whole
    Openable == IF Fixed THEN dirReady
                ELSE \A f \in FileSet : f \in complete /\ writers[f] = 0
end define;

\* complete_job deploys a new student (api.rs:889-893); undeploy removes it (api.rs:759).
\* The Arc the slot held is dropped; the student goes when its last clone does.
fair process deployer = Deployer
variables n = 0;
begin
DNext:-
    while n < Swaps do
        n := n + 1;
        either
            with s \in Students \ used do
                \* (`alive` first: it reads the counts from before this step)
                alive := [t \in Students |->
                            IF t = s THEN TRUE
                            ELSE IF t = slot THEN refs[t] > 1 ELSE alive[t]];
                refs := [Drop(refs, slot) EXCEPT ![s] = 1];
                used := used \cup {s};
                slot := s;
            end with;
        or
            alive := [t \in Students |-> IF t = slot THEN refs[t] > 1 ELSE alive[t]];
            refs := Drop(refs, slot);
            slot := NoStudent;
        end either;
    end while;
end process;

\* classify (api.rs:171-197).
fair process request \in Requests
variables d = NoStudent;
begin
RRead:-     \* p.deployed(): clone the Arc under the read lock, then release it (lib.rs:72-77)
    d := slot;
    if slot # NoStudent then
        refs[slot] := refs[slot] + 1;
    end if;
RPredict:   \* spawn_blocking(predict_proba) and the threshold test, on the clone (api.rs:176-179)
    skip;
RDrop:      \* the clone is dropped with the response
    if d # NoStudent then
        alive[d] := refs[d] > 1;        \* (reads the count from before this step)
        refs[d] := refs[d] - 1;
    end if;
end process;

\* load_student (lib.rs:349-363) for one job.
fair process loader \in Loaders
variables k = 1;
begin
LStart:-    \* a /complete call, or a restart
    skip;
LCheck:     \* `if !dir.join("model.onnx").exists()` (lib.rs:357); after the fix the folder
            \* only ever appears whole
    if InCache then
        goto LOpen;
    end if;
LGet:       \* download_model (store.rs:119-128): each file is read from the store, then written
    while k <= Len(Files) do
        either
            skip;
        or  \* the store read fails: download_model returns the error, load_student fails
            await storeErrors < StoreErrors;
            storeErrors := storeErrors + 1;
            ownError[self] := TRUE;
            result[self] := "failed";
            goto LDone;
        end either;
    LWriteStart:    \* tokio::fs::write creates (truncates) the file ...
        if ~Fixed then      \* after the fix, the temporary folder is this load's own
            writers[Files[k]] := writers[Files[k]] + 1;
            complete := complete \ {Files[k]};
        end if;
    LWriteEnd:      \* ... then writes all of it
        if ~Fixed then
            writers[Files[k]] := writers[Files[k]] - 1;
            complete := complete \cup {Files[k]};
        end if;
        k := k + 1;
    end while;
LRename:    \* after the fix: rename the temporary folder into place; if another load won, use its
    if Fixed then
        dirReady := TRUE;
    end if;
LOpen:      \* Student::load reads the three files (student.rs:63-90)
    if Openable then
        result[self] := "loaded";
    else
        result[self] := "failed";
    end if;
LDone:
    skip;
end process;

end algorithm; *)
\* BEGIN TRANSLATION
VARIABLES slot, refs, alive, used, complete, writers, dirReady, storeErrors, 
          result, ownError, pc

(* define statement *)
Drop(rf, s) == IF s = NoStudent THEN rf ELSE [rf EXCEPT ![s] = @ - 1]

InCache == IF Fixed THEN dirReady
           ELSE "model.onnx" \in complete \/ writers["model.onnx"] > 0

Openable == IF Fixed THEN dirReady
            ELSE \A f \in FileSet : f \in complete /\ writers[f] = 0

VARIABLES n, d, k

vars == << slot, refs, alive, used, complete, writers, dirReady, storeErrors, 
           result, ownError, pc, n, d, k >>

ProcSet == {Deployer} \cup (Requests) \cup (Loaders)

Init == (* Global variables *)
        /\ slot = NoStudent
        /\ refs = [s \in Students |-> 0]
        /\ alive = [s \in Students |-> FALSE]
        /\ used = {}
        /\ complete = {}
        /\ writers = [f \in FileSet |-> 0]
        /\ dirReady = FALSE
        /\ storeErrors = 0
        /\ result = [l \in Loaders |-> "none"]
        /\ ownError = [l \in Loaders |-> FALSE]
        (* Process deployer *)
        /\ n = 0
        (* Process request *)
        /\ d = [self \in Requests |-> NoStudent]
        (* Process loader *)
        /\ k = [self \in Loaders |-> 1]
        /\ pc = [self \in ProcSet |-> CASE self = Deployer -> "DNext"
                                        [] self \in Requests -> "RRead"
                                        [] self \in Loaders -> "LStart"]

DNext == /\ pc[Deployer] = "DNext"
         /\ IF n < Swaps
               THEN /\ n' = n + 1
                    /\ \/ /\ \E s \in Students \ used:
                               /\ alive' = [t \in Students |->
                                              IF t = s THEN TRUE
                                              ELSE IF t = slot THEN refs[t] > 1 ELSE alive[t]]
                               /\ refs' = [Drop(refs, slot) EXCEPT ![s] = 1]
                               /\ used' = (used \cup {s})
                               /\ slot' = s
                       \/ /\ alive' = [t \in Students |-> IF t = slot THEN refs[t] > 1 ELSE alive[t]]
                          /\ refs' = Drop(refs, slot)
                          /\ slot' = NoStudent
                          /\ used' = used
                    /\ pc' = [pc EXCEPT ![Deployer] = "DNext"]
               ELSE /\ pc' = [pc EXCEPT ![Deployer] = "Done"]
                    /\ UNCHANGED << slot, refs, alive, used, n >>
         /\ UNCHANGED << complete, writers, dirReady, storeErrors, result, 
                         ownError, d, k >>

deployer == DNext

RRead(self) == /\ pc[self] = "RRead"
               /\ d' = [d EXCEPT ![self] = slot]
               /\ IF slot # NoStudent
                     THEN /\ refs' = [refs EXCEPT ![slot] = refs[slot] + 1]
                     ELSE /\ TRUE
                          /\ refs' = refs
               /\ pc' = [pc EXCEPT ![self] = "RPredict"]
               /\ UNCHANGED << slot, alive, used, complete, writers, dirReady, 
                               storeErrors, result, ownError, n, k >>

RPredict(self) == /\ pc[self] = "RPredict"
                  /\ TRUE
                  /\ pc' = [pc EXCEPT ![self] = "RDrop"]
                  /\ UNCHANGED << slot, refs, alive, used, complete, writers, 
                                  dirReady, storeErrors, result, ownError, n, 
                                  d, k >>

RDrop(self) == /\ pc[self] = "RDrop"
               /\ IF d[self] # NoStudent
                     THEN /\ alive' = [alive EXCEPT ![d[self]] = refs[d[self]] > 1]
                          /\ refs' = [refs EXCEPT ![d[self]] = refs[d[self]] - 1]
                     ELSE /\ TRUE
                          /\ UNCHANGED << refs, alive >>
               /\ pc' = [pc EXCEPT ![self] = "Done"]
               /\ UNCHANGED << slot, used, complete, writers, dirReady, 
                               storeErrors, result, ownError, n, d, k >>

request(self) == RRead(self) \/ RPredict(self) \/ RDrop(self)

LStart(self) == /\ pc[self] = "LStart"
                /\ TRUE
                /\ pc' = [pc EXCEPT ![self] = "LCheck"]
                /\ UNCHANGED << slot, refs, alive, used, complete, writers, 
                                dirReady, storeErrors, result, ownError, n, d, 
                                k >>

LCheck(self) == /\ pc[self] = "LCheck"
                /\ IF InCache
                      THEN /\ pc' = [pc EXCEPT ![self] = "LOpen"]
                      ELSE /\ pc' = [pc EXCEPT ![self] = "LGet"]
                /\ UNCHANGED << slot, refs, alive, used, complete, writers, 
                                dirReady, storeErrors, result, ownError, n, d, 
                                k >>

LGet(self) == /\ pc[self] = "LGet"
              /\ IF k[self] <= Len(Files)
                    THEN /\ \/ /\ TRUE
                               /\ pc' = [pc EXCEPT ![self] = "LWriteStart"]
                               /\ UNCHANGED <<storeErrors, result, ownError>>
                            \/ /\ storeErrors < StoreErrors
                               /\ storeErrors' = storeErrors + 1
                               /\ ownError' = [ownError EXCEPT ![self] = TRUE]
                               /\ result' = [result EXCEPT ![self] = "failed"]
                               /\ pc' = [pc EXCEPT ![self] = "LDone"]
                    ELSE /\ pc' = [pc EXCEPT ![self] = "LRename"]
                         /\ UNCHANGED << storeErrors, result, ownError >>
              /\ UNCHANGED << slot, refs, alive, used, complete, writers, 
                              dirReady, n, d, k >>

LWriteStart(self) == /\ pc[self] = "LWriteStart"
                     /\ IF ~Fixed
                           THEN /\ writers' = [writers EXCEPT ![Files[k[self]]] = writers[Files[k[self]]] + 1]
                                /\ complete' = complete \ {Files[k[self]]}
                           ELSE /\ TRUE
                                /\ UNCHANGED << complete, writers >>
                     /\ pc' = [pc EXCEPT ![self] = "LWriteEnd"]
                     /\ UNCHANGED << slot, refs, alive, used, dirReady, 
                                     storeErrors, result, ownError, n, d, k >>

LWriteEnd(self) == /\ pc[self] = "LWriteEnd"
                   /\ IF ~Fixed
                         THEN /\ writers' = [writers EXCEPT ![Files[k[self]]] = writers[Files[k[self]]] - 1]
                              /\ complete' = (complete \cup {Files[k[self]]})
                         ELSE /\ TRUE
                              /\ UNCHANGED << complete, writers >>
                   /\ k' = [k EXCEPT ![self] = k[self] + 1]
                   /\ pc' = [pc EXCEPT ![self] = "LGet"]
                   /\ UNCHANGED << slot, refs, alive, used, dirReady, 
                                   storeErrors, result, ownError, n, d >>

LRename(self) == /\ pc[self] = "LRename"
                 /\ IF Fixed
                       THEN /\ dirReady' = TRUE
                       ELSE /\ TRUE
                            /\ UNCHANGED dirReady
                 /\ pc' = [pc EXCEPT ![self] = "LOpen"]
                 /\ UNCHANGED << slot, refs, alive, used, complete, writers, 
                                 storeErrors, result, ownError, n, d, k >>

LOpen(self) == /\ pc[self] = "LOpen"
               /\ IF Openable
                     THEN /\ result' = [result EXCEPT ![self] = "loaded"]
                     ELSE /\ result' = [result EXCEPT ![self] = "failed"]
               /\ pc' = [pc EXCEPT ![self] = "LDone"]
               /\ UNCHANGED << slot, refs, alive, used, complete, writers, 
                               dirReady, storeErrors, ownError, n, d, k >>

LDone(self) == /\ pc[self] = "LDone"
               /\ TRUE
               /\ pc' = [pc EXCEPT ![self] = "Done"]
               /\ UNCHANGED << slot, refs, alive, used, complete, writers, 
                               dirReady, storeErrors, result, ownError, n, d, 
                               k >>

loader(self) == LStart(self) \/ LCheck(self) \/ LGet(self)
                   \/ LWriteStart(self) \/ LWriteEnd(self) \/ LRename(self)
                   \/ LOpen(self) \/ LDone(self)

(* Allow infinite stuttering to prevent deadlock on termination. *)
Terminating == /\ \A self \in ProcSet: pc[self] = "Done"
               /\ UNCHANGED vars

Next == deployer
           \/ (\E self \in Requests: request(self))
           \/ (\E self \in Loaders: loader(self))
           \/ Terminating

Spec == /\ Init /\ [][Next]_vars
        /\ WF_vars((pc[Deployer] # "DNext") /\ deployer)
        /\ \A self \in Requests : WF_vars((pc[self] # "RRead") /\ request(self))
        /\ \A self \in Loaders : WF_vars((pc[self] # "LStart") /\ loader(self))

Termination == <>(\A self \in ProcSet: pc[self] = "Done")

\* END TRANSLATION

-----------------------------------------------------------------------------
(* Properties *)

TypeOK ==
    /\ slot \in Students \cup {NoStudent}
    /\ refs \in [Students -> Nat]
    /\ complete \subseteq FileSet
    /\ result \in [Loaders -> {"none", "loaded", "failed"}]

\* A request never predicts with a student that was dropped (also asserted in RPredict).
InFlightStudentAlive ==
    \A r \in Requests : pc[r] = "RPredict" /\ d[r] # NoStudent => alive[d[r]]

\* A student is dropped once nothing uses it: neither the slot nor a request.
NoLeak ==
    \A s \in Students :
        alive[s] => (slot = s \/ \E r \in Requests : d[r] = s /\ pc[r] \in {"RPredict", "RDrop"})

\* A load fails only because of its own store error, never because of what another load,
\* or an earlier failed one, left in the cache folder.
LoadFailsOnlyOnItsOwnError == \A l \in Loaders : result[l] = "failed" => ownError[l]

=============================================================================
