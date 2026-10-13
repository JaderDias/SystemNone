----------------------------- MODULE AdminConfig -----------------------------
(***************************************************************************)
(* Admin writes to the configuration that /v1/classify reads from memory.  *)
(*                                                                         *)
(* 1. put_settings, put_system_one and delete_system_one write SQLite,     *)
(*    then reload_config reads SQLite into AppState.config_cache           *)
(*    (api.rs:580-581, 643-644, 654-655; lib.rs:239-244).                  *)
(* 2. put_prompt_overrides writes a prompt's overrides to SQLite, then to  *)
(*    PromptState.overrides (api.rs:726-727).                              *)
(*                                                                         *)
(* Each admin request writes its own value (its name), so the properties   *)
(* can tell whose write memory holds. Concurrent PUTs of the settings are  *)
(* last-writer-wins: the admin page always sends every field.              *)
(*                                                                         *)
(* Fixed = FALSE models the code before this change, Fixed = TRUE after:   *)
(* reload_config reads SQLite while it holds the cache's write lock, and   *)
(* put_prompt_overrides holds the overrides' write lock while it writes    *)
(* SQLite.                                                                 *)
(***************************************************************************)
EXTENDS Naturals, FiniteSets, TLC

CONSTANTS
    Admins,   \* concurrent admin requests
    Fixed     \* TRUE: the code with this change's fixes; FALSE: before them

(* --algorithm AdminConfig

variables
    dbSettings = "init",          \* the settings row
    dbModels = "init",            \* the system_one_models table
    cache = <<"init", "init">>,   \* AppState.config_cache: (settings, System One models)
    cacheLock = "free",           \* config_cache's RwLock, held for writing
    dbOv = "init",                \* prompts.overrides of one prompt
    memOv = "init",               \* its PromptState.overrides
    ovLock = "free";              \* PromptState.overrides' RwLock, held for writing

fair process admin \in Admins
variables fresh = <<"init", "init">>;
begin
AChoose:-
    either
        goto ASettings;
    or
        goto AModel;
    or
        goto AOverrides;
    end either;
ASettings:  \* put_settings: db.put_settings (api.rs:580)
    dbSettings := self;
    goto AReload;
AModel:     \* put_system_one / delete_system_one: one write to system_one_models (api.rs:643, 654)
    dbModels := self;
AReload:    \* reload_config (lib.rs:240-243): read SQLite ...
    if Fixed then
        await cacheLock = "free";
        cacheLock := self;
    end if;
    fresh := <<dbSettings, dbModels>>;
ACacheWrite:    \* ... then replace the cache
    cache := fresh;
    if Fixed then cacheLock := "free" end if;
    goto ADone;
AOverrides: \* put_prompt_overrides: db.set_overrides (api.rs:726) ...
    if Fixed then
        await ovLock = "free";
        ovLock := self;
    end if;
    dbOv := self;
AOvMem:     \* ... then *p.overrides.write() = o (api.rs:727)
    memOv := self;
    if Fixed then ovLock := "free" end if;
ADone:
    skip;
end process;

end algorithm; *)
\* BEGIN TRANSLATION
VARIABLES dbSettings, dbModels, cache, cacheLock, dbOv, memOv, ovLock, pc, 
          fresh

vars == << dbSettings, dbModels, cache, cacheLock, dbOv, memOv, ovLock, pc, 
           fresh >>

ProcSet == (Admins)

Init == (* Global variables *)
        /\ dbSettings = "init"
        /\ dbModels = "init"
        /\ cache = <<"init", "init">>
        /\ cacheLock = "free"
        /\ dbOv = "init"
        /\ memOv = "init"
        /\ ovLock = "free"
        (* Process admin *)
        /\ fresh = [self \in Admins |-> <<"init", "init">>]
        /\ pc = [self \in ProcSet |-> "AChoose"]

AChoose(self) == /\ pc[self] = "AChoose"
                 /\ \/ /\ pc' = [pc EXCEPT ![self] = "ASettings"]
                    \/ /\ pc' = [pc EXCEPT ![self] = "AModel"]
                    \/ /\ pc' = [pc EXCEPT ![self] = "AOverrides"]
                 /\ UNCHANGED << dbSettings, dbModels, cache, cacheLock, dbOv, 
                                 memOv, ovLock, fresh >>

ASettings(self) == /\ pc[self] = "ASettings"
                   /\ dbSettings' = self
                   /\ pc' = [pc EXCEPT ![self] = "AReload"]
                   /\ UNCHANGED << dbModels, cache, cacheLock, dbOv, memOv, 
                                   ovLock, fresh >>

AModel(self) == /\ pc[self] = "AModel"
                /\ dbModels' = self
                /\ pc' = [pc EXCEPT ![self] = "AReload"]
                /\ UNCHANGED << dbSettings, cache, cacheLock, dbOv, memOv, 
                                ovLock, fresh >>

AReload(self) == /\ pc[self] = "AReload"
                 /\ IF Fixed
                       THEN /\ cacheLock = "free"
                            /\ cacheLock' = self
                       ELSE /\ TRUE
                            /\ UNCHANGED cacheLock
                 /\ fresh' = [fresh EXCEPT ![self] = <<dbSettings, dbModels>>]
                 /\ pc' = [pc EXCEPT ![self] = "ACacheWrite"]
                 /\ UNCHANGED << dbSettings, dbModels, cache, dbOv, memOv, 
                                 ovLock >>

ACacheWrite(self) == /\ pc[self] = "ACacheWrite"
                     /\ cache' = fresh[self]
                     /\ IF Fixed
                           THEN /\ cacheLock' = "free"
                           ELSE /\ TRUE
                                /\ UNCHANGED cacheLock
                     /\ pc' = [pc EXCEPT ![self] = "ADone"]
                     /\ UNCHANGED << dbSettings, dbModels, dbOv, memOv, ovLock, 
                                     fresh >>

AOverrides(self) == /\ pc[self] = "AOverrides"
                    /\ IF Fixed
                          THEN /\ ovLock = "free"
                               /\ ovLock' = self
                          ELSE /\ TRUE
                               /\ UNCHANGED ovLock
                    /\ dbOv' = self
                    /\ pc' = [pc EXCEPT ![self] = "AOvMem"]
                    /\ UNCHANGED << dbSettings, dbModels, cache, cacheLock, 
                                    memOv, fresh >>

AOvMem(self) == /\ pc[self] = "AOvMem"
                /\ memOv' = self
                /\ IF Fixed
                      THEN /\ ovLock' = "free"
                      ELSE /\ TRUE
                           /\ UNCHANGED ovLock
                /\ pc' = [pc EXCEPT ![self] = "ADone"]
                /\ UNCHANGED << dbSettings, dbModels, cache, cacheLock, dbOv, 
                                fresh >>

ADone(self) == /\ pc[self] = "ADone"
               /\ TRUE
               /\ pc' = [pc EXCEPT ![self] = "Done"]
               /\ UNCHANGED << dbSettings, dbModels, cache, cacheLock, dbOv, 
                               memOv, ovLock, fresh >>

admin(self) == AChoose(self) \/ ASettings(self) \/ AModel(self)
                  \/ AReload(self) \/ ACacheWrite(self) \/ AOverrides(self)
                  \/ AOvMem(self) \/ ADone(self)

(* Allow infinite stuttering to prevent deadlock on termination. *)
Terminating == /\ \A self \in ProcSet: pc[self] = "Done"
               /\ UNCHANGED vars

Next == (\E self \in Admins: admin(self))
           \/ Terminating

Spec == /\ Init /\ [][Next]_vars
        /\ \A self \in Admins : WF_vars((pc[self] # "AChoose") /\ admin(self))

Termination == <>(\A self \in ProcSet: pc[self] = "Done")

\* END TRANSLATION

-----------------------------------------------------------------------------
(* Properties *)

Quiescent == \A a \in Admins : pc[a] \in {"AChoose", "ADone", "Done"}

\* Between requests, the cached settings and System One models are SQLite's.
CacheMatchesDb == Quiescent => cache = <<dbSettings, dbModels>>

\* Between requests, a prompt's overrides in memory are SQLite's.
OverridesMatchDb == Quiescent => memOv = dbOv

=============================================================================
