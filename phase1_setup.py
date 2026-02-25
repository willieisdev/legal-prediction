"""
=============================================================
PHASE 1 — PROJECT SETUP
Predicting U.S. Supreme Court Case Outcomes
=============================================================
Creates folder structure and defines all target mappings
based on the SCDB codebook for caseDisposition codes.
"""

import os

# ── Folder structure ──────────────────────────────────────
DIRS = [
    "data/raw",
    "data/processed",
    "notebooks",
    "models",
    "results/figures",
    "results/tables",
]

for d in DIRS:
    os.makedirs(d, exist_ok=True)

print("✅ Folder structure created:")
for d in DIRS:
    print(f"   {d}/")

# ── caseDisposition codebook mapping ─────────────────────
# Source: SCDB Codebook v02
DISPOSITION_CODES = {
    1:  "stay",
    2:  "affirm",
    3:  "reverse",
    4:  "reverse and remand",
    5:  "vacate and remand",
    6:  "affirm and reverse (split)",
    7:  "vacate",
    8:  "affirm and remand",
    9:  "unspecified",
    10: "dismiss",
    11: "remand",
}

# ── Binary target mapping ─────────────────────────────────
# 0 = Affirm   → codes 2, 8
# 1 = Reverse  → codes 3, 4, 6, 7
# Excluded     → codes 1 (stay), 5 (vacate/remand), 9, 10, 11
#
# Rationale:
#   code 2  = straight affirm             → Affirm (0)
#   code 8  = affirm and remand           → Affirm (0) — lower court upheld
#   code 3  = straight reverse            → Reverse (1)
#   code 4  = reverse and remand          → Reverse (1) — lower court overturned
#   code 6  = affirm and reverse (split)  → Reverse (1) — partial reversal
#   code 7  = vacate                      → Reverse (1) — judgment wiped out
#   code 1  = stay                        → Excluded (procedural)
#   code 5  = vacate and remand           → Remand class only (multiclass)
#   code 9  = unspecified                 → Excluded
#   code 10 = dismiss                     → Excluded (procedural)
#   code 11 = remand only                 → Remand class only (multiclass)

BINARY_MAP = {
    2: 0,   # affirm
    8: 0,   # affirm and remand
    3: 1,   # reverse
    4: 1,   # reverse and remand
    6: 1,   # affirm and reverse (split)
    7: 1,   # vacate
}

BINARY_LABELS = {0: "Affirm", 1: "Reverse"}

# ── Multiclass target mapping ─────────────────────────────
# 0 = Affirm  → codes 2, 8
# 1 = Reverse → codes 3, 6, 7
# 2 = Remand  → codes 4, 5, 11
# 3 = Other   → codes 1, 9, 10  (kept here for multiclass completeness)
#
# Note: codes 1, 9, 10 will be EXCLUDED from binary but kept as
# "Other" in multiclass so students can see model behavior on them.
# If Other class is too small after filtering, it can be dropped.

MULTICLASS_MAP = {
    2:  0,  # affirm
    8:  0,  # affirm and remand
    3:  1,  # reverse
    6:  1,  # affirm and reverse (split)
    7:  1,  # vacate
    4:  2,  # reverse and remand
    5:  2,  # vacate and remand
    11: 2,  # remand
    1:  3,  # stay       → Other
    9:  3,  # unspecified → Other
    10: 3,  # dismiss     → Other
}

MULTICLASS_LABELS = {0: "Affirm", 1: "Reverse", 2: "Remand", 3: "Other"}

# ── Pre-decision features (safe to use) ───────────────────
PRE_DECISION_FEATURES = [
    "term",                  # year of term (integer)
    "naturalCourt",          # justice lineup code
    "jurisdiction",          # how case reached SCOTUS
    "certReason",            # why cert was granted
    "issueArea",             # broad legal issue (1–14)
    "petitioner",            # petitioner type code
    "respondent",            # respondent type code
    "lcDisposition",         # lower court ruling code
    "lcDispositionDirection",# liberal(1) / conservative(2) direction of LC ruling
    "lcDisagreement",        # was there dissent below? (0/1)
    "threeJudgeFdc",         # three-judge district court? (0/1)
    "caseOrigin",            # originating court code
    "caseSource",            # immediate source court code
    "adminAction",           # administrative agency action code
    "decisionType",          # type of decision (opinion, per curiam, etc.)
    "authorityDecision1",    # primary legal authority
    "lawType",               # type of law at issue
    "petitionerState",       # state of petitioner (sparse)
    "respondentState",       # state of respondent (sparse)
    "caseOriginState",       # state of origin court (sparse)
    "caseSourceState",       # state of source court (sparse)
]

# ── Post-decision leakage variables (MUST EXCLUDE) ────────
LEAKAGE_VARS = [
    "majVotes",               # vote count — known only after decision
    "minVotes",               # vote count — known only after decision
    "voteUnclear",            # post-decision flag
    "declarationUncon",       # post-decision ruling detail
    "partyWinning",           # literally the outcome
    "decisionDirection",      # post-decision ideological direction
    "decisionDirectionDissent",
    "precedentAlteration",    # post-decision flag
    "splitVote",              # constant (all 1s) — no predictive value
    "majOpinWriter",          # assigned after decision
    "majOpinAssigner",        # assigned after decision
    "caseDisposition",        # THE TARGET — exclude from features
    "caseDispositionUnusual", # post-decision annotation
    "authorityDecision2",     # mostly missing + post-decision context
]

# ── Columns to drop entirely (IDs / citations / dates) ────
DROP_COLS = [
    "caseId", "docketId", "caseIssuesId", "voteId",
    "usCite", "sctCite", "ledCite", "lexisCite",
    "docket", "caseName", "chief",
    "dateDecision", "dateArgument", "dateRearg",
    "issue",       # too granular (10000+ codes); issueArea is sufficient
    "lawSupp",     # statute supplement — too sparse / granular
    "lawMinor",    # string field, very sparse
    "adminActionState",  # too sparse (663 non-null)
]

# ── Chronological split year ───────────────────────────────
TRAIN_END_YEAR  = 2005   # inclusive
TEST_START_YEAR = 2006   # inclusive

# ── Random seed ────────────────────────────────────────────
RANDOM_STATE = 42

print("\n✅ Target mappings defined:")
print(f"   Binary classes   : {BINARY_LABELS}")
print(f"   Multiclass labels: {MULTICLASS_LABELS}")
print(f"   Pre-decision features ({len(PRE_DECISION_FEATURES)}): listed in PRE_DECISION_FEATURES")
print(f"   Leakage vars to remove ({len(LEAKAGE_VARS)}): listed in LEAKAGE_VARS")
print(f"\n✅ Train years: 1946 – {TRAIN_END_YEAR}")
print(f"✅ Test  years: {TEST_START_YEAR} – 2024")
print("\n✅ Phase 1 complete. Import this module in all subsequent phases.")
