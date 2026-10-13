------------------------------ MODULE InputLog ------------------------------
(***************************************************************************)
(* The buffered input log: /v1/classify pushes each input into a buffer,   *)
(* and flushes write the buffer to the store, one object per prompt. Three *)
(* tasks flush: the background task (on a timer, or when the batch is      *)
(* full), the claim handler before it hands out a job ("the worker must    *)
(* see every logged input"), and main after the server stops.              *)
(*                                                                         *)
(* Code: gateway/src/store.rs (InputLog::push, InputLog::flush),           *)
(* gateway/src/lib.rs (spawn_background, flush_all), gateway/src/api.rs    *)
(* (classify, claim_job) and gateway/src/main.rs (run).                    *)
(*                                                                         *)
(* Fixed = FALSE models the code before this change, Fixed = TRUE after:   *)
(* a flush then holds an async mutex until its writes are done.            *)
(***************************************************************************)
EXTENDS Naturals, FiniteSets, Sequences, TLC

CONSTANTS
    Prompts,        \* prompts the inputs belong to
    Requests,       \* /v1/classify requests; each logs one input (named after the request)
    Claims,         \* POST /internal/jobs/claim calls
    Bg,             \* the background flush task
    Main,           \* main, which flushes once more after the server stops
    Fixed,          \* TRUE: the code with this change's fixes; FALSE: before them
    WriteFailures   \* how many store writes may fail while the server runs

(* --algorithm InputLog

variables
    buf = [p \in Prompts |-> {}],       \* InputLog.buf
    written = [r \in Requests |-> 0],   \* how many times the store holds each input
    flushLock = "free",                 \* after the fix: InputLog's flush mutex
    failures = 0,
    stopping = FALSE,                   \* the server stopped taking requests
    exited = FALSE,                     \* main returned: the runtime dropped every other task
    \* bookkeeping for the properties
    pushed = {},                        \* inputs in the buffer (the push returned)
    claimSawAll = TRUE;                 \* every claim saw the inputs logged before it

\* InputLog::flush (store.rs:171-201). A task dropped by the runtime stops where it
\* stands, hence `await ~exited` in each step.
procedure Flush()
variables taken = [p \in Prompts |-> {}], todo = {};
begin
FLock:
    await ~exited;
    if Fixed then
        await flushLock = "free";
        flushLock := self;
    end if;
FTake:      \* std::mem::take(&mut *self.buf.lock()) (store.rs:172-173)
    await ~exited;
    taken := buf;
    todo := {p \in Prompts : buf[p] # {}};
    buf := [p \in Prompts |-> {}];
FWrite:     \* one self.store.put(&key, ...).await per prompt (store.rs:176-199)
    while todo # {} do
        await ~exited;
        with p \in todo do
            todo := todo \ {p};
            either
                written := [r \in Requests |->
                              IF r \in taken[p] THEN written[r] + 1 ELSE written[r]];
            or  \* the store refuses it: the records go back to the buffer for the next flush
                await failures < WriteFailures /\ ~stopping;
                failures := failures + 1;
                buf[p] := buf[p] \cup taken[p];
            end either;
        end with;
    end while;
FUnlock:
    if Fixed then flushLock := "free" end if;
    return;
end procedure;

\* /v1/classify: s.inputs.push(&p.id, &req.input) (api.rs:167-169, store.rs:153-168).
fair process request \in Requests
begin
RPush:-
    await ~stopping;
    with p \in Prompts do
        buf[p] := buf[p] \cup {self};
    end with;
    pushed := pushed \cup {self};
end process;

\* spawn_background (lib.rs:167-179): flush every flush_secs, or when woken by a full batch.
fair process background = Bg
begin
BLoop:
    while TRUE do
        call Flush();
    end while;
end process;

\* claim_job (api.rs:801-818): s.flush_all().await, then the worker lists the store.
fair process claim \in Claims
variables mustSee = {}, failuresBefore = 0;
begin
CStart:-
    await ~stopping;
    mustSee := pushed;
    failuresBefore := failures;
    call Flush();
CCheck:
    claimSawAll := claimSawAll /\
        (failures = failuresBefore => \A r \in mustSee : written[r] > 0);
end process;

\* main.rs:95-98: graceful shutdown waits for the requests in flight, then flush_all,
\* then main returns and the runtime drops the background task. Strong fairness: tokio's
\* Mutex is first come, first served, so the looping background task can't starve main.
fair+ process main = Main
begin
MSignal:-
    stopping := TRUE;
MDrain:
    await \A c \in Claims : pc[c] \in {"CStart", "Done"};
MFlush:
    call Flush();
MExit:
    exited := TRUE;
end process;

end algorithm; *)
\* BEGIN TRANSLATION
VARIABLES buf, written, flushLock, failures, stopping, exited, pushed, 
          claimSawAll, pc, stack, taken, todo, mustSee, failuresBefore

vars == << buf, written, flushLock, failures, stopping, exited, pushed, 
           claimSawAll, pc, stack, taken, todo, mustSee, failuresBefore >>

ProcSet == (Requests) \cup {Bg} \cup (Claims) \cup {Main}

Init == (* Global variables *)
        /\ buf = [p \in Prompts |-> {}]
        /\ written = [r \in Requests |-> 0]
        /\ flushLock = "free"
        /\ failures = 0
        /\ stopping = FALSE
        /\ exited = FALSE
        /\ pushed = {}
        /\ claimSawAll = TRUE
        (* Procedure Flush *)
        /\ taken = [ self \in ProcSet |-> [p \in Prompts |-> {}]]
        /\ todo = [ self \in ProcSet |-> {}]
        (* Process claim *)
        /\ mustSee = [self \in Claims |-> {}]
        /\ failuresBefore = [self \in Claims |-> 0]
        /\ stack = [self \in ProcSet |-> << >>]
        /\ pc = [self \in ProcSet |-> CASE self \in Requests -> "RPush"
                                        [] self = Bg -> "BLoop"
                                        [] self \in Claims -> "CStart"
                                        [] self = Main -> "MSignal"]

FLock(self) == /\ pc[self] = "FLock"
               /\ ~exited
               /\ IF Fixed
                     THEN /\ flushLock = "free"
                          /\ flushLock' = self
                     ELSE /\ TRUE
                          /\ UNCHANGED flushLock
               /\ pc' = [pc EXCEPT ![self] = "FTake"]
               /\ UNCHANGED << buf, written, failures, stopping, exited, 
                               pushed, claimSawAll, stack, taken, todo, 
                               mustSee, failuresBefore >>

FTake(self) == /\ pc[self] = "FTake"
               /\ ~exited
               /\ taken' = [taken EXCEPT ![self] = buf]
               /\ todo' = [todo EXCEPT ![self] = {p \in Prompts : buf[p] # {}}]
               /\ buf' = [p \in Prompts |-> {}]
               /\ pc' = [pc EXCEPT ![self] = "FWrite"]
               /\ UNCHANGED << written, flushLock, failures, stopping, exited, 
                               pushed, claimSawAll, stack, mustSee, 
                               failuresBefore >>

FWrite(self) == /\ pc[self] = "FWrite"
                /\ IF todo[self] # {}
                      THEN /\ ~exited
                           /\ \E p \in todo[self]:
                                /\ todo' = [todo EXCEPT ![self] = todo[self] \ {p}]
                                /\ \/ /\ written' = [r \in Requests |->
                                                       IF r \in taken[self][p] THEN written[r] + 1 ELSE written[r]]
                                      /\ UNCHANGED <<buf, failures>>
                                   \/ /\ failures < WriteFailures /\ ~stopping
                                      /\ failures' = failures + 1
                                      /\ buf' = [buf EXCEPT ![p] = buf[p] \cup taken[self][p]]
                                      /\ UNCHANGED written
                           /\ pc' = [pc EXCEPT ![self] = "FWrite"]
                      ELSE /\ pc' = [pc EXCEPT ![self] = "FUnlock"]
                           /\ UNCHANGED << buf, written, failures, todo >>
                /\ UNCHANGED << flushLock, stopping, exited, pushed, 
                                claimSawAll, stack, taken, mustSee, 
                                failuresBefore >>

FUnlock(self) == /\ pc[self] = "FUnlock"
                 /\ IF Fixed
                       THEN /\ flushLock' = "free"
                       ELSE /\ TRUE
                            /\ UNCHANGED flushLock
                 /\ pc' = [pc EXCEPT ![self] = Head(stack[self]).pc]
                 /\ taken' = [taken EXCEPT ![self] = Head(stack[self]).taken]
                 /\ todo' = [todo EXCEPT ![self] = Head(stack[self]).todo]
                 /\ stack' = [stack EXCEPT ![self] = Tail(stack[self])]
                 /\ UNCHANGED << buf, written, failures, stopping, exited, 
                                 pushed, claimSawAll, mustSee, failuresBefore >>

Flush(self) == FLock(self) \/ FTake(self) \/ FWrite(self) \/ FUnlock(self)

RPush(self) == /\ pc[self] = "RPush"
               /\ ~stopping
               /\ \E p \in Prompts:
                    buf' = [buf EXCEPT ![p] = buf[p] \cup {self}]
               /\ pushed' = (pushed \cup {self})
               /\ pc' = [pc EXCEPT ![self] = "Done"]
               /\ UNCHANGED << written, flushLock, failures, stopping, exited, 
                               claimSawAll, stack, taken, todo, mustSee, 
                               failuresBefore >>

request(self) == RPush(self)

BLoop == /\ pc[Bg] = "BLoop"
         /\ stack' = [stack EXCEPT ![Bg] = << [ procedure |->  "Flush",
                                                pc        |->  "BLoop",
                                                taken     |->  taken[Bg],
                                                todo      |->  todo[Bg] ] >>
                                            \o stack[Bg]]
         /\ taken' = [taken EXCEPT ![Bg] = [p \in Prompts |-> {}]]
         /\ todo' = [todo EXCEPT ![Bg] = {}]
         /\ pc' = [pc EXCEPT ![Bg] = "FLock"]
         /\ UNCHANGED << buf, written, flushLock, failures, stopping, exited, 
                         pushed, claimSawAll, mustSee, failuresBefore >>

background == BLoop

CStart(self) == /\ pc[self] = "CStart"
                /\ ~stopping
                /\ mustSee' = [mustSee EXCEPT ![self] = pushed]
                /\ failuresBefore' = [failuresBefore EXCEPT ![self] = failures]
                /\ stack' = [stack EXCEPT ![self] = << [ procedure |->  "Flush",
                                                         pc        |->  "CCheck",
                                                         taken     |->  taken[self],
                                                         todo      |->  todo[self] ] >>
                                                     \o stack[self]]
                /\ taken' = [taken EXCEPT ![self] = [p \in Prompts |-> {}]]
                /\ todo' = [todo EXCEPT ![self] = {}]
                /\ pc' = [pc EXCEPT ![self] = "FLock"]
                /\ UNCHANGED << buf, written, flushLock, failures, stopping, 
                                exited, pushed, claimSawAll >>

CCheck(self) == /\ pc[self] = "CCheck"
                /\ claimSawAll' = (           claimSawAll /\
                                   (failures = failuresBefore[self] => \A r \in mustSee[self] : written[r] > 0))
                /\ pc' = [pc EXCEPT ![self] = "Done"]
                /\ UNCHANGED << buf, written, flushLock, failures, stopping, 
                                exited, pushed, stack, taken, todo, mustSee, 
                                failuresBefore >>

claim(self) == CStart(self) \/ CCheck(self)

MSignal == /\ pc[Main] = "MSignal"
           /\ stopping' = TRUE
           /\ pc' = [pc EXCEPT ![Main] = "MDrain"]
           /\ UNCHANGED << buf, written, flushLock, failures, exited, pushed, 
                           claimSawAll, stack, taken, todo, mustSee, 
                           failuresBefore >>

MDrain == /\ pc[Main] = "MDrain"
          /\ \A c \in Claims : pc[c] \in {"CStart", "Done"}
          /\ pc' = [pc EXCEPT ![Main] = "MFlush"]
          /\ UNCHANGED << buf, written, flushLock, failures, stopping, exited, 
                          pushed, claimSawAll, stack, taken, todo, mustSee, 
                          failuresBefore >>

MFlush == /\ pc[Main] = "MFlush"
          /\ stack' = [stack EXCEPT ![Main] = << [ procedure |->  "Flush",
                                                   pc        |->  "MExit",
                                                   taken     |->  taken[Main],
                                                   todo      |->  todo[Main] ] >>
                                               \o stack[Main]]
          /\ taken' = [taken EXCEPT ![Main] = [p \in Prompts |-> {}]]
          /\ todo' = [todo EXCEPT ![Main] = {}]
          /\ pc' = [pc EXCEPT ![Main] = "FLock"]
          /\ UNCHANGED << buf, written, flushLock, failures, stopping, exited, 
                          pushed, claimSawAll, mustSee, failuresBefore >>

MExit == /\ pc[Main] = "MExit"
         /\ exited' = TRUE
         /\ pc' = [pc EXCEPT ![Main] = "Done"]
         /\ UNCHANGED << buf, written, flushLock, failures, stopping, pushed, 
                         claimSawAll, stack, taken, todo, mustSee, 
                         failuresBefore >>

main == MSignal \/ MDrain \/ MFlush \/ MExit

Next == background \/ main
           \/ (\E self \in ProcSet: Flush(self))
           \/ (\E self \in Requests: request(self))
           \/ (\E self \in Claims: claim(self))

Spec == /\ Init /\ [][Next]_vars
        /\ \A self \in Requests : WF_vars((pc[self] # "RPush") /\ request(self))
        /\ WF_vars(background) /\ WF_vars(Flush(Bg))
        /\ \A self \in Claims : WF_vars((pc[self] # "CStart") /\ claim(self)) /\ WF_vars(Flush(self))
        /\ SF_vars((pc[Main] # "MSignal") /\ main) /\ SF_vars(Flush(Main))

\* END TRANSLATION

-----------------------------------------------------------------------------
(* Properties *)

TypeOK ==
    /\ buf \in [Prompts -> SUBSET Requests]
    /\ written \in [Requests -> Nat]
    /\ pushed \subseteq Requests

\* No input is written to the store twice.
NoDuplicates == \A r \in Requests : written[r] <= 1

\* When a claim's flush returns, the store holds every input logged before the claim
\* started (unless a store write failed meanwhile: those records wait for the next flush).
ClaimSeesEarlierInputs == claimSawAll

\* Once main has returned, every logged input is in the store.
NoLossOnShutdown == exited => \A r \in pushed : written[r] > 0

\* Every logged input is eventually written (the background task keeps flushing).
InputsEventuallyWritten == \A r \in Requests : r \in pushed ~> written[r] > 0

=============================================================================
