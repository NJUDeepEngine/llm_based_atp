import Lean.Message
import Lean.Elab.Tactic
import Lean.Elab.Frontend

open Lean Lean.Meta Lean.Elab Lean.Elab.Command Lean.Elab.Tactic

namespace LeanDojo

private def printResponse {α : Type _} [ToJson α] (res : α) : IO Unit := do
  let json := (toJson res).pretty 99999999999999999
  println! "REPL> {json}"
  (← IO.getStdout).flush

private def join (l : List String) (sep : String := "\n") : String :=
  match l with
  | [] => ""
  | first :: others =>
    others.foldl (fun r s => r ++ sep ++ s) first

structure Request where
  sid: Nat
  cmd: String
deriving FromJson, ToJson

structure Response where
  sid : Option Nat := none
  tacticState : Option String := none
  error: Option String := none
deriving ToJson

structure ReplState (σ : Type _) where
  savedStates : Array σ
  savedStatesStrs : Array String
  solvedState : Option σ

private def getSavedState? (m : Type → Type) [Monad m] {σ : Type _} [MonadState (ReplState σ) m] (sid : Nat) : m (Option σ) := do
  return (← get).savedStates[sid]?

private def getInitialState! (m : Type → Type) [Monad m] {σ : Type _} [MonadState (ReplState σ) m] [MonadError m] : m σ := do
  let some ts ← getSavedState? m 0
    | throwError "[fatal] no initial state"
  return ts

private def getNextSid (m : Type → Type) [Monad m] {σ : Type _} [MonadState (ReplState σ) m] : m Nat := do
  return (← get).savedStates.size

namespace TacticRepl


abbrev TacticReplM := StateT (ReplState Tactic.SavedState) TacticM

instance : MonadLift IO TacticReplM where
  monadLift x := liftM x

def ppTacticState (ts : Tactic.SavedState) : TacticM String := do
    match ts.tactic.goals with
    | [] => return "no goals"
    | [g] => return (← Meta.ppGoal g).pretty
    | goals =>
      return (← goals.foldlM (fun a b => do return a ++ "\n\n" ++ (← Meta.ppGoal b).pretty) "").trim

private def insertTacticState (ts : Tactic.SavedState) : TacticReplM Unit := do
  let succeeded := ts.tactic.goals.isEmpty
  let ts_str ← ppTacticState ts
  modifyGet fun s => ((), ⟨s.savedStates.push ts, s.savedStatesStrs.push ts_str,
    match s.solvedState with
    | some _ => s.solvedState
    | none => if succeeded then ts else none
  ⟩)

private def initializeTacticRepl : TacticM Tactic.SavedState := do
  if not (← isProp (← getMainTarget)) then
    throwError "[fatal] not_a_theorem"
  pruneSolvedGoals
  let ts ← Tactic.saveState
  let ts_str ← ppTacticState ts
  let res : Response := {sid := some 0, tacticState := ts_str}
  printResponse res
  return ts

private def levels2Names : List Level → NameSet
  | [] => NameSet.empty
  | Level.param n :: us => (levels2Names us).insert n
  | _ :: us => levels2Names us

private def collectFromLevel : Level → NameSet
| Level.zero => NameSet.empty
| Level.succ l => collectFromLevel l
| Level.param n => NameSet.empty.insert n
| Level.max l1 l2 => (collectFromLevel l1).union $ collectFromLevel l2
| Level.imax l1 l2 => (collectFromLevel l1).union $ collectFromLevel l2
| Level.mvar _ => NameSet.empty

private def collectLevelParams : Expr → NameSet
  | .sort u => collectFromLevel u
  | .const _ us => levels2Names us
  | .app fm arg => (collectLevelParams fm).union $ collectLevelParams arg
  | .lam _ binderType body _ => (collectLevelParams binderType).union $ collectLevelParams body
  | .forallE _ binderType body _ => (collectLevelParams binderType).union $ collectLevelParams body
  | .letE _ type value body _ => ((collectLevelParams type).union $ collectLevelParams value).union $ collectLevelParams body
  | .mdata _ expr => collectLevelParams expr
  | .proj _ _ struct => collectLevelParams struct
  | _ => NameSet.empty

private def collectFVarsAux : Expr → NameSet
  | .fvar fvarId => NameSet.empty.insert fvarId.name
  | .app fm arg => (collectFVarsAux fm).union $ collectFVarsAux arg
  | .lam _ binderType body _ => (collectFVarsAux binderType).union $ collectFVarsAux body
  | .forallE _ binderType body _ => (collectFVarsAux binderType).union $ collectFVarsAux body
  | .letE _ type value body _ => ((collectFVarsAux type).union $ collectFVarsAux value).union $ collectFVarsAux body
  | .mdata _ expr => collectFVarsAux expr
  | .proj _ _ struct => collectFVarsAux struct
  | _ => NameSet.empty

private def collectFVars (e : Expr) : MetaM (Array Expr) := do
  let names := collectFVarsAux e
  let mut fvars := #[]
  for ldecl in ← getLCtx do
    if ldecl.isImplementationDetail then
      continue
    if names.contains ldecl.fvarId.name then
      fvars := fvars.push $ .fvar ldecl.fvarId
  return fvars

private def abstractAllLambdaFVars (e : Expr) : MetaM Expr := do
  let mut e' := e
  while e'.hasFVar do
    let fvars ← collectFVars e'
    if fvars.isEmpty then
      break
    e' ← mkLambdaFVars fvars e'
  return e'

private def getSidByState? (ts'_str : String) : TacticReplM (Option Nat) := do
  let savedStatesStrs := (← get).savedStatesStrs
  for (ts_str, tsid) in savedStatesStrs.zipWithIndex do
    if ts_str == ts'_str then
      return some tsid
  return none

private def validateProof : TacticReplM Response := do
  let ts ← Tactic.saveState

  let ts0 ← getInitialState! TacticReplM
  ts0.restore
  let [goalId] ← getGoals | throwError "[fatal] more than one initial goal"
  let tgt ← getMainTarget >>= instantiateMVars
  let tgt_fmt ← ppExpr tgt

  ts.restore
  let some pf ← getExprMVarAssignment? goalId | throwError "[fatal] goal not assigned"
  let pf ← instantiateMVars pf
  let pft ← inferType pf >>= instantiateMVars
  let pft_fmt ← ppExpr pft

  if ! (← withTransparency .all (isExprDefEq tgt pft)) then
    return {error := s!"proof type mismatch: {tgt_fmt} != {pft_fmt}"}

  ts0.restore
  let pf ← goalId.withContext $ abstractAllLambdaFVars pf
  let pft ← inferType pf >>= instantiateMVars

  ts.restore
  if pf.hasSorry then
    return {error := "proof contains `sorry`"}

  if pf.hasExprMVar then
    return {error := "proof contains metavariables"}

  let lvls := (collectLevelParams pf).toList
  let decl := Declaration.thmDecl {
      name := Name.anonymous, type := pft, value := pf
      levelParams := lvls
  }
  try
    let _ ← addDecl decl
  catch ex =>
    return {error := s!"kernel type check failed: {← ex.toMessageData.toString}"}

  -- let ts_str ← ppTacticState ts
  -- insertTacticState ts
  -- let sid ← match ← getSidByState? ts_str with
  --         | some index => pure index
  --         | none => do
  --             let next_tsid ← getNextSid TacticReplM
  --             insertTacticState ts
  --             pure next_tsid
  -- return {sid := sid, tacticState := ts_str}
  let ts_str ← ppTacticState ts
  let next_tsid ← getNextSid TacticReplM
  insertTacticState ts
  return {sid := next_tsid, tacticState := ts_str}

private def handleRunTac (req : Request) : TacticReplM Response := do
  match ← getSavedState? TacticReplM req.sid with
  | none => throwError s!"[fatal] unknown tsid: {req.sid}"
  | some ts =>
    match Parser.runParserCategory (← getEnv) `tactic req.cmd "<stdin>" with
    | .error err => return {error := err}
    | .ok stx =>
      ts.restore
      try
        monadLift $ commitIfNoEx (evalTactic stx)
        let s ← getThe Core.State
        if s.messages.hasErrors then
          let messages := s.messages.toList.filter fun m => m.severity == MessageSeverity.error
          return { error := join $ ← (messages.map (·.data)).mapM fun md => md.toString }
      catch ex =>
        return {error := ← ex.toMessageData.toString}

      pruneSolvedGoals
      if (← getGoals).isEmpty then
        validateProof
      else
        let ts' ← Tactic.saveState
        let ts'_str ← ppTacticState ts'
        let sid ← match ← getSidByState? ts'_str with
          | some index => pure index
          | none => do
              let next_tsid ← getNextSid TacticReplM
              insertTacticState ts'
              pure next_tsid

        return {sid := sid, tacticState := ts'_str}


end TacticRepl


private def loop (m : Type → Type) [Monad m] [MonadLift IO m] [MonadError m] (handler : Request → m Response) : m Unit := do
 while true do
    -- 读取用户输入的一行命令
    let line := (← (← IO.getStdin).getLine).trim
    if line == "exit" then
      break  -- 退出循环
    -- 解析JSON格式的请求
    match (Json.parse line) with
    | .error err => throwError s!"[fatal] failed to parse JSON {err}"
    | .ok cmd =>
      match (fromJson? cmd : Except String Request) with
      | .error err => throwError s!"[fatal] parse_failed: data={err}"
      | .ok req => (← handler req) |> printResponse  -- 处理请求并打印响应


namespace TacticRepl

/--
示例交互流程：
{"sid": 0, "cmd": "skip"}
{"sid": 1, "cmd": "rw [add_assoc, add_comm b, ←add_assoc]"}
exit
--/
def repl : TacticM Unit := do
  withMainContext do
    -- 打印初始目标
    let ts ← initializeTacticRepl
    let ts_str ← ppTacticState ts
    -- 启动交互循环
    let loop := LeanDojo.loop TacticReplM handleRunTac
    let (_, s) ← loop.run {savedStates := #[ts], savedStatesStrs := #[ts_str], solvedState := none}
    -- 若存在解决的战术状态，恢复并退出
    match s.solvedState with
    | none => return ()
    | some ts' => ts'.restore
  IO.Process.exit 0

end TacticRepl

end LeanDojo


/-- The `lean_dojo_repl` tactic. --/
elab "lean_dojo_repl" : tactic => LeanDojo.TacticRepl.repl
