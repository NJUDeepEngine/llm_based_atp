import Lean
import Lake
open Lean Elab System

set_option maxHeartbeats 2000000

instance : ToJson Substring where
  toJson s := toJson s.toString

instance : ToJson String.Pos where
  toJson n := toJson n.1

deriving instance ToJson for SourceInfo
deriving instance ToJson for Syntax.Preresolved
deriving instance ToJson for Syntax
deriving instance ToJson for Position

namespace LeanDojo

structure Trace where
  commandASTs : Array Syntax
deriving ToJson

namespace Path

def relativeTo (path parent : FilePath) : Option FilePath :=
  let rec componentsRelativeTo (pathComps parentComps : List String) : Option FilePath :=
    match pathComps, parentComps with
    | _, [] => mkFilePath pathComps
    | [], _ => none
    | (h₁ :: t₁), (h₂ :: t₂) =>
      if h₁ == h₂ then
        componentsRelativeTo t₁ t₂
      else
        none

    componentsRelativeTo path.components parent.components

def isRelativeTo (path parent : FilePath) : Bool :=
  match relativeTo path parent with
  | some _ => true
  | none => false

def toAbsolute (path : FilePath) : IO FilePath := do
  if path.isAbsolute then
    pure path
  else
    let cwd ← IO.currentDir
    pure $ cwd / path

private def trim (path : FilePath) : FilePath :=
  assert! path.isRelative
  mkFilePath $ path.components.filter (· != ".")

def packagesDir : FilePath := ".lake/packages"

def buildDir : FilePath := ".lake/build"

def libDir : FilePath := buildDir / "lib"

def toBuildDir (subDir : FilePath) (path : FilePath) (ext : String) : Option FilePath :=
  let path' := (trim path).withExtension ext
  return buildDir / subDir / path'

def makeParentDirs (p : FilePath) : IO Unit := do
  let some parent := p.parent | throw $ IO.userError s!"Unable to get the parent of {p}"
  IO.FS.createDirAll parent

end Path

def canProcess (path : FilePath): IO Bool := do
  assert! path.isAbsolute
  if !(← path.pathExists) then
    return false
  if (← path.isDir) ∨ path.extension != "lean" then
    return false
  let cwd ← IO.currentDir
  let some relativePath := Path.relativeTo path cwd |
    throw $ IO.userError s!"Invalid path: {path}"
  if Path.isRelativeTo relativePath Path.packagesDir then
    return false
  let some oleanPath := Path.toBuildDir "lib" relativePath "olean" |
    throw $ IO.userError s!"Invalid path: {path}"
  return ← oleanPath.pathExists

unsafe def processFile (path : String) : IO Unit := do
  let path ← Path.toAbsolute ⟨path⟩
  assert! (← canProcess path)
  let input ← IO.FS.readFile path
  enableInitializersExecution
  let inputCtx := Parser.mkInputContext input path.toString
  let (header, parserState, messages) ← Parser.parseHeader inputCtx
  let (env, messages) ← processHeader header {} messages inputCtx
  if messages.hasErrors then
    for msg in messages.toList do
      if msg.severity == .error then
        println! "ERROR: {← msg.toString}"
    throw $ IO.userError "Errors during import; aborting"
  let env := env.setMainModule (← moduleNameOfFileName path none)
  let commandState := { Command.mkState env messages {} with infoState.enabled := true }
  let s ← IO.processCommands inputCtx parserState commandState
  let commands := s.commands.pop -- Remove EOI command.
  let trace : Trace := ⟨#[header] ++ commands⟩
  let cwd ← IO.currentDir
  assert! cwd.fileName != "lean4"
  let some relativePath := Path.relativeTo path cwd | throw $ IO.userError s!"Invalid path: {path}"
  let json_path := Path.toBuildDir "ir" relativePath "ast.json" |>.get!
  Path.makeParentDirs json_path
  IO.FS.writeFile json_path (toJson trace).pretty

end LeanDojo

open LeanDojo
unsafe def main (args : List String) : IO Unit := do
  match args with
  | [] =>
    println! "Please provide at least one file!"
  | paths =>
    for path in paths do
      processFile path
