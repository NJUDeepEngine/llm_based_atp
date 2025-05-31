import re
from pathlib import Path

_LEAN4_VERSION_REGEX = re.compile(r"leanprover/lean4:(?P<version>.+?)")
_SINGLE_LINE_COMMENT_REGEX = r"--.*?(\n|$)" 
_MULTI_LINE_COMMENT_REGEX = r"/-.*?(-/|$)"  
_COMMENT_REGEX = re.compile(  
    f"{_SINGLE_LINE_COMMENT_REGEX}|{_MULTI_LINE_COMMENT_REGEX}", 
    re.DOTALL  
)
_CAMEL_CASE_REGEX = re.compile(r"(_|-)+")
DECL_KW = r"(?:example|lemma|theorem)"
BLOCK_RE = re.compile(
    rf"""
    (?P<block>                                  
        ^[ \t]*{DECL_KW}\s+                     
        (?P<name>\w+)                          
        (?:\s*[:=])?                            
        [\s\S]*?                                
        \bby\b                                  
        [\s\S]*?                                
        (?:                                     
            \bsorry\b                          
          |                                     
            (?=^[ \t]*{DECL_KW}\b)              
        )
    )
    """,
    re.MULTILINE | re.VERBOSE,
)

LEAN4_PACKAGES_DIR = Path(".lake/packages")
LEAN4_BUILD_DIR = Path(".lake/build")
LEAN4_DATA_EXTRACTOR_PATH = Path(__file__).parent / "data_extraction" / "ExtractData.lean"
LEAN4_REPL_PATH = Path(__file__).parent / "interaction" / "Lean4Repl.lean"
assert LEAN4_DATA_EXTRACTOR_PATH.exists() and LEAN4_REPL_PATH.exists()