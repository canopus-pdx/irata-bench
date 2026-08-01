# PEP 563: defer annotation evaluation. Several functions (e.g.
# run_pregame_interview) annotate `engine: Engine` before class Engine is
# defined further down the file. Python 3.14+ tolerates this because PEP 649
# made annotations lazy by default, but on 3.13 and older it raises NameError
# at import time. This line makes the file run on Python 3.7+ regardless.
from __future__ import annotations

import abc
import json
import logging
import os
import random
import sys
import textwrap
import time
from datetime import datetime

# API SDKs
from anthropic import Anthropic
from google import genai
from google.genai import types
from openai import OpenAI

# ==========================================
# 0. BENCHMARK CONFIGURATION TOGGLES & VERSION
# ==========================================
BENCHMARK_VERSION = "2.7.1"  # Holt-Laury calibration, inter-match memos, capital-filtered EV, CoT-first, hypocrisy/capital metrics
ENABLE_RANDOM_EVENTS = False  # Disabled for pure deterministic strategy data!
POVERTY_CASH_THRESHOLD = 25  # Cash threshold below which Dan Tax is waived ($0)
BASE_LAND_CLAIM_FEE = 25  # Base fee to claim any land tile (stops free land-banking!)
HISTORY_LOG_MAX_ENTRIES = 12  # Cap player decision memory to last N entries to prevent context bloat & timeouts

# ==========================================
# 0-V27. V2.7.0 BEHAVIORAL-ECONOMICS FEATURES
# ==========================================
# Holt-Laury risk calibration: a one-time 5-question lottery task per model,
# run before any game state exists, producing a static Risk Preference Score
# (0-5). Higher switch point = more risk-averse. Stored permanently and
# correlated against in-game inaction. Costs ~5 tiny calls once, then never again.
ENABLE_HOLT_LAURY = True

# Inter-match memo: at match end a model may leave a one-line private directive
# to its FUTURE self, injected into its next pre-game turn as a self-instruction.
# Tests whether self-authored commitment can override trained hesitation.
ENABLE_INTER_MATCH_MEMO = True

# Capital-filtered EV-positive pass tracking. A pass only counts as "tax
# paralysis" if the player COULD have afforded to act (cash >= Dan Tax) AND
# selling would have been net-positive. Passing at $7 cash needing energy is
# arithmetic, not fear — those are excluded.
TRACK_EV_POSITIVE_PASSES = True

# CoT-first schema: when True, decision prompts ask for reasoning in a bounded
# "reasoning" field BEFORE the action, and the action field is emitted so it can
# never be truncated. Directly fixes the mid-JSON truncation seen when a model
# writes a long note_to_self and the token budget cuts off before the close brace.
COT_FIRST_SCHEMA = True

# Hypocrisy & capital-depletion metrics, computed post-match from data we already
# capture. Separates "said aggressive, played passive" (hypocrisy) from "ran out
# of cash in round 3" (capital bankruptcy) — distinguishable by cash trajectory.
TRACK_HYPOCRISY_AND_CAPITAL = True

# ==========================================
# 0-V26. V2.6.0 FEATURE TOGGLES
# ==========================================
# Baseline (control) agents. Without a control group, "model X won a 4-player
# game" only proves it was less chaotic than three peers — not that it is
# competent. RandomAgent (pure random legal moves) proves models beat noise;
# GreedyAgent (simple heuristic) sets the floor for basic economic awareness.
# Set the roster below to include or exclude them.
#   "FRONTIER"  -> the four LLMs (classic)
#   "MIXED"     -> swap DeepSeek for Greedy + Random so a match still has 4 seats
#   "BASELINES" -> Random vs Greedy only (a quick sanity/calibration match)
ROSTER_MODE = "FRONTIER"  # "FRONTIER" | "MIXED" | "BASELINES"

# Board seeding. Terrain RNG was a major confounder: a lucky river cluster is a
# structural advantage unrelated to skill. A fixed seed lets every model face
# identical terrain for true A/B isolation. None => fresh random board each match.
BOARD_SEED = None  # e.g. 20260727 to replay one specific board

# Compact state serialization. Abbreviating development codes and stripping
# redundant board keys cuts token overhead ~30-50%, lowering latency and 504s
# structurally rather than just widening the timeout. Set False for verbose
# (human-debuggable) payloads.
COMPACT_STATE_PAYLOAD = True

# Persist the most human-readable outputs (pre/post interviews) and a structured
# round-by-round decision ledger into the match record, so Strategic Consistency
# (did the model do what it said it would?) becomes measurable after the fact.
LOG_VERBATIM_INTERVIEWS = True
LOG_STRUCTURED_DECISIONS = True

# Mid-game net worth snapshots: net worth per player at the end of each round's
# production phase (value created, before selling). Unlocks trajectory analysis
# and the end-of-match timeline chart.
SNAPSHOT_NET_WORTH_PER_ROUND = True
RENDER_TIMELINE_CHART = True  # Write an SVG net-worth-over-time chart per match

# Player memory: before each match, give every contestant its OWN history —
# past interview texts, prior decision patterns, stated-vs-actual strategy, and
# performance broken down by rule set (high tax, few rounds, etc.). Without this
# each pre-game interview is amnesia: a model says "I've recalibrated" with no
# actual recollection of what it did. This makes reflection genuine and lets us
# test whether models adapt across matches. Capped to avoid context bloat.
GIVE_PLAYERS_MEMORY = True
MEMORY_MAX_MATCHES = 4  # how many recent own-matches to summarize in detail

# Secret self-messages: a contestant may attach an optional "note_to_self" to any
# decision. It is fed back ONLY to that same player on future turns — never to
# opponents — and recorded for the operator (you) to read. This is a private
# scratchpad for multi-turn planning ("hoard crystite, sell round 12"), and it
# makes strategic intent readable instead of merely inferred.
ENABLE_SECRET_NOTES = True
SECRET_NOTES_MAX = 8          # how many recent notes to feed back to the author
SECRET_NOTE_MAX_CHARS = 300   # truncate to keep the payload lean

# ==========================================
# 0-VR. VARIABLE-RULES BENCHMARK (V2.5.0)
# ==========================================
# The fixed-rules benchmark converged: three of four models became statistically
# indistinguishable. A model that only ever plays 12-round / 5%-tax / trade-every-4
# M.U.L.E. can win by executing ONE memorized optimal strategy. That measures
# recall of a fixed optimum, not general business competence.
#
# V2.5.0 draws a fresh, bounded-random rule set for every match and DISCLOSES it
# to every player at match start (it is in the state payload on turn one). This
# tests planning under known-but-varying constraints — the cleanest form of the
# business question "who adapts best when the rules change each quarter?"
#
# IMPORTANT: rules are fixed for the whole match and known from round 1. This is
# deliberately NOT a mid-game-surprise test (that is a separate, harder experiment).
ENABLE_VARIABLE_RULES = True  # False => use the FIXED_* defaults below (old 2.4.0 behaviour)

# Fixed fallbacks / defaults (used when ENABLE_VARIABLE_RULES is False)
FIXED_TOTAL_ROUNDS = 12
FIXED_TRADE_INTERVAL = 4
FIXED_DAN_TAX_PERCENTAGE = 0.05
FIXED_STARTING_CASH = 300
FIXED_SCARCITY_THRESHOLDS = {"FOOD": 8, "ENERGY": 8, "CRYSTITE": 4}

# Randomization bounds (inclusive). Tuned to stay within safe operating limits:
# rounds capped at 20 to bound context-window growth and latency (Claude already
# brushes the 15s ceiling); tax capped at 10%; cash kept in a range that never
# trivializes the $25 land fee.
VR_ROUNDS_MIN, VR_ROUNDS_MAX = 10, 20
VR_TRADE_INTERVAL_MIN, VR_TRADE_INTERVAL_MAX = 2, 5
VR_DAN_TAX_MIN, VR_DAN_TAX_MAX = 0.00, 0.10
VR_STARTING_CASH_MIN, VR_STARTING_CASH_MAX = 250, 400
VR_SCARCITY_JITTER = 0.20  # ±20% on the store scarcity thresholds
# The Dan Tax base can also vary, per your "source of the taxes change" idea.
# NET_WORTH reproduces 2.4.0 (taxes total wealth, so hoarders pay more).
# CASH_ON_HAND taxes only liquid cash, so it punishes sitting on cash and
# rewards converting it into tiles/resources — a genuinely different incentive.
# Both are charged at trade time from state that already exists, so no
# chicken-and-egg with the trade value. (TRADE_VALUE was considered but dropped:
# taxing a trade before it is proposed is ill-defined.)
VR_TAX_BASIS_OPTIONS = ["NET_WORTH", "CASH_ON_HAND"]

# ==========================================
# 0a. FAIRNESS: TIME BUDGET & SDK RETRIES
# ==========================================
# V2.3.0 advertised a "universal 10-second ceiling" that was not universal.
# The Anthropic and OpenAI SDKs default to max_retries=2 and apply the timeout
# PER ATTEMPT, so those agents silently got 3 x 10s ≈ 31s of wall clock while
# Gemini's http_options deadline is a hard total. Observed in real logs:
# Claude 31,606ms vs Gemini 9,653ms against the same nominal budget.
# Setting retries to 0 makes the ceiling mean the same thing for everyone.
SDK_MAX_RETRIES = 0

# 15s rather than 10s: gemini-3.6-flash is a reasoning model and was losing
# turns by ~250ms (real timeouts at 9,653ms and 9,738ms). Applied to ALL agents.
API_TIMEOUT_SECONDS = 20.0

# ==========================================
# 0b. SCORING: RESOURCE VALUES vs STORE PRICES
# ==========================================
# These MUST differ. In V2.3.0 every resource scored at exactly its base store
# price, so selling returned the same value as holding, minus the Dan Tax.
# Liquidating was therefore strictly negative-EV and the optimal final-round
# move was to do nothing. Real runs confirmed it: Gemini ended matches holding
# ~10.8 crystite (tax-free) while every other model sold and paid the tax.
# A business benchmark should not reward refusing to transact.
#
# Scoring values now sit ~15% BELOW base store prices, so liquidating carries a
# genuine spread, while the Dan Tax still makes small/frequent trades wasteful.
# That produces a real timing decision instead of a dominant do-nothing move.
RESOURCE_SCORE_VALUES = {"FOOD": 12, "ENERGY": 20, "CRYSTITE": 85}
STORE_BASE_PRICES = {"FOOD": 15, "ENERGY": 25, "CRYSTITE": 100}
STORE_SCARCITY_PRICES = {"FOOD": 30, "ENERGY": 45, "CRYSTITE": 150}
DEVELOPED_TILE_VALUE = 50
UNDEVELOPED_TILE_VALUE = 25

# ==========================================
# 0c. OPERATIONAL HEALTH PENALTY
# ==========================================
# "linear" reproduces V2.3.0: max(floor, 1 - 0.02*B - 0.05*S). That saturates —
# 10 food shortages alone hit the 0.5 floor, after which blackouts were FREE.
# A model could deliberately walk off that cliff (and one did: the archived
# starve-and-hoard run reached 46 blackouts at zero marginal cost).
# "multiplicative" charges a real marginal cost for every single failure and
# never fully saturates.
PENALTY_MODE = "multiplicative"  # "multiplicative" | "linear"
BLACKOUT_PENALTY_RATE = 0.02
SHORTAGE_PENALTY_RATE = 0.05
PENALTY_FLOOR = 0.10

# ==========================================
# 0d. MODEL REGISTRY  ← CHANGE MODEL NAMES HERE, NOWHERE ELSE
# ==========================================
# Google retires Gemini models aggressively (2.0-flash died 2026-06-01,
# 2.5-flash shortly after, 2.5-flash-lite ~2026-07-22). Hardcoding a single ID
# guarantees a future 404. Instead we keep an ordered preference list and let
# the agent pick the first one the API actually reports as available.
GEMINI_MODEL_PREFERENCES = [
    "gemini-3.6-flash",       # GA 2026-07-21, current workhorse Flash
    "gemini-3.5-flash-lite",  # GA 2026-07-21, fastest / cheapest 3.5-class
    "gemini-3.5-flash",       # GA, behind the gemini-flash-latest alias
    "gemini-flash-latest",    # Floating alias — never 404s, but drifts
]
CLAUDE_MODEL = "claude-sonnet-4-6"
OPENAI_MODEL = "gpt-4o"
DEEPSEEK_MODEL = "deepseek-chat"

# ==========================================
# 0c. DIAGNOSTICS & RELIABILITY TOGGLES
# ==========================================
RUN_PREFLIGHT_CHECK = True    # Ping every API before the match starts
ABORT_ON_PREFLIGHT_FAIL = False  # True = refuse to run if any agent is dead
PREFLIGHT_TIMEOUT_SECONDS = 15.0  # Preflight gets a longer leash than in-game calls
DEGRADED_AGENT_THRESHOLD = 3  # Consecutive failures before an agent is flagged DEGRADED
TIMESTAMPED_LOG_FILES = True  # False = always write plain game_run.log (overwrites)


# ==========================================
# 0d. MATCH CONFIG — the per-match rule set
# ==========================================
class MatchConfig:
  """The rule set for a single match.

  Every parameter that used to be a module-level global (rounds, trade cadence,
  tax rate, starting cash, store scarcity) now lives here so it can vary per
  match while staying fixed and fully known for the duration of that match.

  The engine reads from an instance of this instead of from globals, and the
  same instance is serialized both into the state payload (so players can plan)
  and into the match record (so results are reproducible)."""

  def __init__(
      self,
      total_rounds,
      trade_interval,
      dan_tax_percentage,
      starting_cash,
      scarcity_thresholds,
      tax_basis="NET_WORTH",
      variable_rules=False,
      seed=None,
  ):
    self.total_rounds = total_rounds
    self.trade_interval = trade_interval          # store + direct trades share cadence
    self.dan_tax_percentage = dan_tax_percentage
    self.starting_cash = starting_cash
    self.scarcity_thresholds = dict(scarcity_thresholds)
    self.tax_basis = tax_basis
    self.variable_rules = variable_rules
    self.seed = seed

  @property
  def trade_rounds(self):
    """The concrete round numbers on which trading occurs."""
    return [r for r in range(1, self.total_rounds + 1)
            if r % self.trade_interval == 0]

  @classmethod
  def fixed(cls):
    """The classic 2.4.0 rule set."""
    return cls(
        total_rounds=FIXED_TOTAL_ROUNDS,
        trade_interval=FIXED_TRADE_INTERVAL,
        dan_tax_percentage=FIXED_DAN_TAX_PERCENTAGE,
        starting_cash=FIXED_STARTING_CASH,
        scarcity_thresholds=FIXED_SCARCITY_THRESHOLDS,
        tax_basis="NET_WORTH",
        variable_rules=False,
        seed=None,
    )

  @classmethod
  def random_bounded(cls, seed=None):
    """Draw a fresh, bounded rule set. If seed is given the draw is reproducible,
    which is what makes a variable-rules match re-runnable from its record."""
    rng = random.Random(seed)
    jitter = lambda base: max(1, round(base * (1 + rng.uniform(-VR_SCARCITY_JITTER,
                                                               VR_SCARCITY_JITTER))))
    return cls(
        total_rounds=rng.randint(VR_ROUNDS_MIN, VR_ROUNDS_MAX),
        trade_interval=rng.randint(VR_TRADE_INTERVAL_MIN, VR_TRADE_INTERVAL_MAX),
        dan_tax_percentage=round(rng.uniform(VR_DAN_TAX_MIN, VR_DAN_TAX_MAX), 3),
        starting_cash=rng.randint(VR_STARTING_CASH_MIN, VR_STARTING_CASH_MAX),
        scarcity_thresholds={
            "FOOD": jitter(FIXED_SCARCITY_THRESHOLDS["FOOD"]),
            "ENERGY": jitter(FIXED_SCARCITY_THRESHOLDS["ENERGY"]),
            "CRYSTITE": jitter(FIXED_SCARCITY_THRESHOLDS["CRYSTITE"]),
        },
        tax_basis=rng.choice(VR_TAX_BASIS_OPTIONS),
        variable_rules=True,
        seed=seed,
    )

  @classmethod
  def for_new_match(cls, seed=None):
    return cls.random_bounded(seed) if ENABLE_VARIABLE_RULES else cls.fixed()

  def tax_basis_description(self):
    return {
        "NET_WORTH": "your total net worth at trade time",
        "CASH_ON_HAND": "your cash on hand at trade time",
        "TRADE_VALUE": "the gross value of the trade being proposed",
    }.get(self.tax_basis, self.tax_basis)

  def to_dict(self):
    """For the state payload and the match record."""
    return {
        "total_rounds": self.total_rounds,
        "trade_interval": self.trade_interval,
        "trade_rounds": self.trade_rounds,
        "dan_tax_percentage": self.dan_tax_percentage,
        "starting_cash": self.starting_cash,
        "store_scarcity_thresholds": self.scarcity_thresholds,
        "dan_tax_basis": self.tax_basis,
        "variable_rules_enabled": self.variable_rules,
        "seed": self.seed,
    }

  def disclosure_text(self):
    """Human-readable rules block injected into every decision prompt so players
    can plan against the ACTUAL rules of THIS match rather than assumed defaults."""
    lines = [
        "MATCH RULES (fixed for this entire match, known to all players):",
        f"- This match lasts EXACTLY {self.total_rounds} rounds.",
        f"- Trading occurs on rounds: {self.trade_rounds} "
        f"(every {self.trade_interval} rounds).",
        f"- Every player started with ${self.starting_cash} cash.",
    ]
    if self.dan_tax_percentage <= 0:
      lines.append("- Dan Tax: NONE this match — trades are free.")
    else:
      lines.append(
          f"- Dan Tax: {self.dan_tax_percentage:.1%} of "
          f"{self.tax_basis_description()}, charged on every trade attempt "
          "(waived if you are nearly broke)."
      )
    lines.append(
        f"- Store scarcity kicks in below these stock levels: "
        f"{self.scarcity_thresholds} (prices rise when stock is scarce)."
    )
    if self.variable_rules:
      lines.append(
          "- NOTE: these parameters are randomized per match. Plan for THESE "
          "numbers, not the standard 12-round game."
      )
    return "\n".join(lines)


# ==========================================
# 1. LOGGING SETUP (WITH MILLISECOND TIMESTAMPS)
# ==========================================
_RUN_STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# Previously mode="w" on a fixed filename, which destroyed the prior run's log
# every time — so a regression could never be compared against the run before it.
if TIMESTAMPED_LOG_FILES:
  os.makedirs("logs", exist_ok=True)
  MAIN_LOG_PATH = os.path.join("logs", f"game_run_{_RUN_STAMP}.log")
  ERROR_LOG_PATH = os.path.join("logs", f"errors_{_RUN_STAMP}.log")
  # Convenience symlink/copy target so "game_run.log" always means "the latest run"
  LATEST_LOG_PATH = "game_run.log"
else:
  MAIN_LOG_PATH = "game_run.log"
  ERROR_LOG_PATH = "errors.log"
  LATEST_LOG_PATH = None

file_formatter = logging.Formatter(
    "%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

class InfraOnlyFilter(logging.Filter):
  """Gates the errors-only log to genuine infrastructure faults.

  V2.3.0 attached the error handler at WARNING level, but BLACKOUT and
  WORK SHORTAGE are also logged at WARNING — so the file meant to answer
  'what broke?' filled up with routine gameplay. A real run came out 12/20
  lines of blackouts. Those events are benchmark *signal*, not system faults:
  they belong in the main log, not the incident log.

  Infrastructure call sites opt in with extra={"infra": True}."""

  def filter(self, record):
    return getattr(record, "infra", False)


# Marker to pass to logger calls that describe infrastructure problems.
INFRA = {"infra": True}

file_handler = logging.FileHandler(MAIN_LOG_PATH, mode="w")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(file_formatter)

# Errors-only log: when something breaks, this file is short enough to read at a glance
error_handler = logging.FileHandler(ERROR_LOG_PATH, mode="w")
error_handler.setLevel(logging.WARNING)
error_handler.setFormatter(file_formatter)
error_handler.addFilter(InfraOnlyFilter())

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter("📢 %(message)s")
console_handler.setFormatter(console_formatter)

logger = logging.getLogger("IrataBench")
logger.setLevel(logging.DEBUG)
logger.addHandler(file_handler)
logger.addHandler(error_handler)
logger.addHandler(console_handler)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


def finalize_logs():
  """Copies the timestamped run log to game_run.log so the newest run is always
  at a predictable path (handy for sharing/uploading)."""
  if not LATEST_LOG_PATH:
    return
  try:
    for h in logger.handlers:
      h.flush()
    with open(MAIN_LOG_PATH, "r") as src, open(LATEST_LOG_PATH, "w") as dst:
      dst.write(src.read())
  except Exception as e:
    logger.debug(f"Could not mirror log to {LATEST_LOG_PATH}: {e}")

# ==========================================
# 1b. API ERROR TAXONOMY
# ==========================================
# Every agent failure used to log the same generic line: "Timed out or failed".
# That line was identical whether the key was revoked, the model was retired, we
# were rate-limited, or the network actually timed out — four completely
# different fixes. This classifier separates them and attaches the fix.

ERROR_CATEGORIES = [
    "AUTH",             # Bad/revoked/missing credentials
    "MODEL_NOT_FOUND",  # Model ID retired or misspelled
    "RATE_LIMIT",       # Quota / throttling
    "TIMEOUT",          # Genuine slow response or client-side deadline
    "BAD_REQUEST",      # Malformed params (wrong units, bad config field)
    "SERVER_ERROR",     # Provider-side 5xx
    "UNKNOWN",
]


def classify_api_error(exc: Exception, duration_ms: int) -> tuple[str, str]:
  """Maps an SDK exception to (category, actionable_hint).

  duration_ms matters: a 'timeout' that returns in 200ms is never a real
  network timeout — it is a misconfigured client-side deadline or an
  instant rejection. That distinction cost us two debugging rounds."""
  text = str(exc).lower()

  # --- Credentials ---
  if any(k in text for k in (
      "401", "403", "unauthenticated", "permission_denied", "api key not valid",
      "invalid api key", "api_key_invalid", "unauthorized", "authentication",
  )):
    return "AUTH", (
        "Credential rejected. Check the relevant *_API_KEY env var is exported, "
        "not expired, and not revoked (keys committed to source get auto-revoked)."
    )

  # --- Retired / misspelled model ---
  if any(k in text for k in (
      "404", "not_found", "not found", "no longer available",
      "is not found for api version", "does not exist",
  )):
    return "MODEL_NOT_FOUND", (
        "Model ID is retired or misspelled. Update the MODEL REGISTRY at the top "
        "of this file. Run with RUN_PREFLIGHT_CHECK=True to list live model IDs."
    )

  # --- Quota ---
  if any(k in text for k in (
      "429", "resource_exhausted", "rate limit", "quota", "too many requests",
  )):
    return "RATE_LIMIT", (
        "Rate limited / out of quota. Slow the match down, or check billing on "
        "the provider dashboard."
    )

  # --- Malformed request ---
  if any(k in text for k in (
      "400", "invalid_argument", "bad request", "unexpected keyword",
      "validation error", "unknown field",
  )):
    return "BAD_REQUEST", (
        "Provider rejected the request shape. Usually a config field the SDK "
        "version does not accept, or a unit mismatch (seconds vs milliseconds)."
    )

  # --- Provider-side failure ---
  if any(k in text for k in (
      "500", "502", "503", "internal", "unavailable", "overloaded", "bad gateway",
  )):
    return "SERVER_ERROR", (
        "Provider-side error. Usually transient — retry, or the service is "
        "having an incident."
    )

  # --- Timeouts (last, because the word is overloaded) ---
  if any(k in text for k in (
      "timeout", "timed out", "deadline_exceeded", "504", "read timed out",
  )):
    if duration_ms < 1000:
      return "BAD_REQUEST", (
          f"Reported a timeout after only {duration_ms}ms — far too fast to be a "
          "real network timeout. The client-side deadline is almost certainly "
          "misconfigured (check seconds-vs-milliseconds in http_options)."
      )
    return "TIMEOUT", (
        f"Genuine timeout after {duration_ms}ms against a "
        f"{API_TIMEOUT_SECONDS}s ceiling. Either the model is too slow for this "
        "budget or the prompt is too large — try a faster model."
    )

  return "UNKNOWN", "Unrecognized error shape. See the full exception text above."


# ==========================================
# 2. PERMANENT MATCH HISTORY, VERSION SYNC & STATS
# ==========================================
HISTORY_FILE = "match_history.json"
CHECKPOINT_FILE = "turn_checkpoint.json"


def _empty_history_file():
  """Writes a fresh v2.2.0 schema skeleton to disk."""
  skeleton = {
      "schema_version": BENCHMARK_VERSION,
      "generated_at": datetime.now().strftime("%Y-%m-%d"),
      "notes": [
          "Matches 1-4 predate V2.0.0 and are missing benchmark_version and telemetry_metrics fields.",
          "Dan Tax figures are logged exactly per trade starting from V2.2.0.",
          "Win streak and peak stats are recomputed from all matches on every save.",
      ],
      "all_time_stats": {},
      "matches": [],
  }
  with open(HISTORY_FILE, "w") as f:
      json.dump(skeleton, f, indent=2)


def load_match_history() -> list:
  """Loads the matches list from the v2.2.0 wrapped schema.
  Falls back gracefully if the file is missing, empty, or in the old bare-list format."""
  if not os.path.exists(HISTORY_FILE):
    try:
      _empty_history_file()
    except Exception as e:
      logger.error(f"Failed to create new match history file: {e}")
    return []
  try:
    with open(HISTORY_FILE, "r") as f:
      content = f.read().strip()
    if not content:
      return []
    data = json.loads(content)
    # New wrapped format: {"schema_version": ..., "matches": [...]}
    if isinstance(data, dict) and "matches" in data:
      return data["matches"]
    # Legacy bare-list format (pre-v2.2.0): migrate transparently on next save
    if isinstance(data, list):
      logger.debug("[HISTORY] Legacy bare-list format — will migrate to v2.2.1 schema on next save.")
      return data
    return []
  except Exception as e:
    logger.error(f"Error loading match history: {e}, resetting history.", exc_info=True)
    return []


def load_full_history_document() -> dict:
  """Loads the full JSON document (wrapper + matches). Returns a fresh skeleton if missing."""
  if not os.path.exists(HISTORY_FILE):
    return {"schema_version": BENCHMARK_VERSION, "all_time_stats": {}, "matches": []}
  try:
    with open(HISTORY_FILE, "r") as f:
      content = f.read().strip()
    if not content:
      return {"schema_version": BENCHMARK_VERSION, "all_time_stats": {}, "matches": []}
    data = json.loads(content)
    if isinstance(data, dict) and "matches" in data:
      return data
    if isinstance(data, list):
      return {"schema_version": BENCHMARK_VERSION, "all_time_stats": {}, "matches": data}
    return {"schema_version": BENCHMARK_VERSION, "all_time_stats": {}, "matches": []}
  except Exception as e:
    logger.error(f"Error loading full history document: {e}", exc_info=True)
    return {"schema_version": BENCHMARK_VERSION, "all_time_stats": {}, "matches": []}


def verify_version_sync():
  """Verifies engine version against history ledger at startup."""
  logger.info(f"🔄 [VERSION CHECK] Launching Irata-Bench Engine V{BENCHMARK_VERSION}")
  doc = load_full_history_document()
  matches = doc.get("matches", [])
  if matches:
    latest = matches[-1]
    last_ver = latest.get("benchmark_version", "pre-2.0")
    logger.info(f"📜 [VERSION CHECK] Last recorded match was saved under V{last_ver}")
    if last_ver != BENCHMARK_VERSION:
      logger.info(f"✨ [VERSION NOTICE] Version upgrade detected (V{last_ver} -> V{BENCHMARK_VERSION}). System ready!")
  file_schema = doc.get("schema_version", "legacy")
  if file_schema != BENCHMARK_VERSION:
    logger.info(f"📋 [SCHEMA NOTICE] File schema V{file_schema} will be migrated to V{BENCHMARK_VERSION} on next save.")


def get_player_win_loss_stats(player_name: str) -> tuple[int, int]:
  """Calculates total wins and losses for a specific player from match_history.json."""
  history = load_match_history()
  wins = 0
  losses = 0
  for match in history:
    winner = match.get("winner")
    scores = match.get("final_scores", {})
    if player_name in scores:
      if winner == player_name:
        wins += 1
      else:
        losses += 1
  return wins, losses


def get_compact_history_summary() -> dict:
  doc = load_full_history_document()
  matches = doc.get("matches", [])
  all_time = doc.get("all_time_stats", {})

  if not matches:
    return {"total_matches": 0, "standings": "No prior matches recorded."}

  # Prefer pre-computed all_time_stats wins if available (avoids a full scan)
  if all_time and "players" in all_time:
    win_counts = {p: all_time["players"][p]["wins"] for p in all_time["players"]}
  else:
    win_counts = {}
    for match in matches:
      winner = match.get("winner", "Unknown")
      win_counts[winner] = win_counts.get(winner, 0) + 1

  recent_matches = [
      {
          "match_id": m.get("match_id"),
          "winner": m.get("winner"),
          "scores": m.get("final_scores"),
      }
      for m in matches[-2:]
  ]

  return {
      "total_matches_played": len(matches),
      "all_time_wins": win_counts,
      "recent_matches": recent_matches,
  }


def get_player_memory_dossier(player_name: str) -> dict:
  """Builds a player's PERSONAL memory from the match history: its own past
  interviews, what it actually did, whether its stated strategy matched its
  behavior, and how it has performed under different rule sets.

  This is per-player and self-only — a model sees its OWN track record, not a
  god's-eye view of opponents' internals. Returns a compact dict suitable for
  injecting into the pre-game prompt."""
  doc = load_full_history_document()
  matches = doc.get("matches", [])
  mine = [m for m in matches if player_name in m.get("final_scores", {})]
  if not mine:
    return {"note": "No prior matches — this is your debut."}

  # ---- Per-match personal recaps (most recent MEMORY_MAX_MATCHES) ----
  recaps = []
  for m in mine[-MEMORY_MAX_MATCHES:]:
    cfg = m.get("match_config", {})
    fs = m.get("final_scores", {})
    my_score = fs.get(player_name, 0)
    winner = m.get("winner")
    won = winner == player_name
    ledger = m.get("decision_ledger", {}).get(player_name, [])
    claims = sum(1 for e in ledger if "CLAIM" in e)
    trades = sum(1 for e in ledger if ("TRADE" in e or "SELL" in e or "BUY" in e))
    passes = sum(1 for e in ledger if e.endswith("PASS") or ": PASS" in e)
    stats = m.get("player_stats", {}).get(player_name, {})
    pre = m.get("interviews", {}).get("pregame", {}).get(player_name, "")
    post = m.get("interviews", {}).get("postgame", {}).get(player_name, "")

    # What the winner did differently (only the headline, not their internals).
    winner_stats = m.get("player_stats", {}).get(winner, {})
    recaps.append({
        "match_id": m.get("match_id"),
        "rules": (
            f"{cfg.get('total_rounds','?')} rounds, "
            f"{cfg.get('dan_tax_percentage',0):.1%} tax on "
            f"{cfg.get('dan_tax_basis','?')}, "
            f"trade every {cfg.get('trade_interval','?')}, "
            f"${cfg.get('starting_cash','?')} start"
        ),
        "result": f"{'WON' if won else 'lost'} — you scored {my_score}, "
                  f"winner was {winner} ({fs.get(winner,0)})",
        "what_you_did": (
            f"{claims} tiles claimed, {trades} trade actions, {passes} passes; "
            f"ended holding {stats.get('crystite',0)} crystite, "
            f"{stats.get('food',0)} food, {stats.get('energy',0)} energy, "
            f"${stats.get('cash',0)} cash; "
            f"{stats.get('total_blackouts',0)} blackouts, "
            f"{stats.get('food_shortages',0)} food shortages"
        ),
        "what_winner_held": (
            f"{winner} ended with {winner_stats.get('crystite',0)} crystite, "
            f"${winner_stats.get('cash',0)} cash, "
            f"{winner_stats.get('tiles_owned','?')} tiles"
        ) if not won else "you won this one",
        "you_said_before": pre,
        "you_said_after": post,
    })

  # ---- Performance conditioned on rule set (the transferable lesson) ----
  def bucket(m):
    cfg = m.get("match_config", {})
    tax = cfg.get("dan_tax_percentage", 0)
    rounds = cfg.get("total_rounds", 12)
    tax_lvl = "high_tax" if tax >= 0.06 else ("low_tax" if tax <= 0.02 else "mid_tax")
    len_lvl = "long_game" if rounds >= 16 else ("short_game" if rounds <= 12 else "mid_game")
    basis = cfg.get("dan_tax_basis", "NET_WORTH")
    return tax_lvl, len_lvl, basis

  by_condition = {}
  for m in mine:
    fs = m.get("final_scores", {})
    won = m.get("winner") == player_name
    ranked = sorted(fs.values(), reverse=True)
    my_rank = ranked.index(fs[player_name]) + 1 if fs.get(player_name) in ranked else None
    for key in bucket(m):
      d = by_condition.setdefault(key, {"matches": 0, "wins": 0, "ranks": []})
      d["matches"] += 1
      d["wins"] += 1 if won else 0
      if my_rank:
        d["ranks"].append(my_rank)

  condition_summary = {}
  for key, d in by_condition.items():
    avg_rank = round(sum(d["ranks"]) / len(d["ranks"]), 2) if d["ranks"] else None
    condition_summary[key] = (
        f"{d['wins']}/{d['matches']} wins, avg finish {avg_rank}"
    )

  wins = sum(1 for m in mine if m.get("winner") == player_name)
  return {
      "your_record": f"{wins} wins / {len(mine) - wins} losses over {len(mine)} matches",
      "recent_match_recaps": recaps,
      "performance_by_rule_set": condition_summary,
      "coaching_note": (
          "Review your past interviews against what you actually did. If you "
          "promised aggression and then passed repeatedly, adjust. Note which "
          "rule sets you win in and which you struggle in."
      ),
  }


def save_turn_checkpoint(engine):
  checkpoint_data = {
      "benchmark_version": BENCHMARK_VERSION,
      "current_round": engine.current_round,
      "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
      "scores": {p.name: p.net_worth(engine) for p in engine.players},
      "player_stats": {
          p.name: {
              "cash": p.cash,
              "food": p.food,
              "energy": p.energy,
              "crystite": p.crystite,
              "tiles_owned": len(p.owned_tiles),
          }
          for p in engine.players
      },
  }
  try:
    with open(CHECKPOINT_FILE, "w") as f:
      json.dump(checkpoint_data, f, indent=2)
  except Exception as e:
    logger.error(f"Checkpoint save failed: {e}")


def _recompute_all_time_stats(matches: list) -> dict:
  """Recomputes the full all_time_stats block from scratch over all matches."""
  # Derive the full set of seat names actually seen, so baseline rosters
  # (Random/Greedy) and any renamed seats aggregate correctly rather than being
  # silently dropped. Falls back to the classic four if the file is empty.
  seen = []
  for m in matches:
    for name in m.get("final_scores", {}):
      if name not in seen:
        seen.append(name)
  players = seen if seen else ["Gemini", "Claude", "ChatGPT", "DeepSeek"]

  wins   = {p: 0 for p in players}
  losses = {p: 0 for p in players}

  peak_cash      = {p: {"value": 0, "match_id": None} for p in players}
  peak_energy    = {p: {"value": 0, "match_id": None} for p in players}
  peak_food      = {p: {"value": 0, "match_id": None} for p in players}
  peak_crystite  = {p: {"value": 0, "match_id": None} for p in players}
  peak_score     = {p: {"value": 0, "match_id": None} for p in players}
  peak_tiles     = {p: {"value": 0, "match_id": None} for p in players}

  total_blackouts   = {p: 0 for p in players}
  total_shortages   = {p: 0 for p in players}
  total_mule_fail   = {p: 0 for p in players}
  total_timeouts    = {p: 0 for p in players}
  total_api_calls   = {p: 0 for p in players}

  trades_proposed             = {p: 0 for p in players}
  trades_accepted_as_proposer = {p: 0 for p in players}
  trades_received             = {p: 0 for p in players}
  trades_accepted_as_receiver = {p: 0 for p in players}
  total_dan_tax_paid          = {p: 0 for p in players}

  current_streaks = {p: 0 for p in players}
  best_streak     = {p: {"length": 0, "ended_at_match": None} for p in players}

  biggest_margin = {"value": 0, "winner": None, "match_id": None}
  closest_match  = {"gap": 999999, "match_id": None}
  most_trades_match   = {"count": 0, "match_id": None}
  richest_match  = {"total_wealth": 0, "match_id": None}
  poorest_match  = {"total_wealth": 9999999, "match_id": None}

  for m in matches:
    mid    = m["match_id"]
    winner = m["winner"]
    final  = m.get("final_scores", {})
    stats  = m.get("player_stats", {})
    trades = m.get("trade_summary", [])
    telem  = m.get("telemetry_metrics", {})

    for p in players:
      if p not in final:
        continue

      if winner == p:
        wins[p] += 1
        current_streaks[p] += 1
        if current_streaks[p] > best_streak[p]["length"]:
          best_streak[p] = {"length": current_streaks[p], "ended_at_match": mid}
      else:
        losses[p] += 1
        current_streaks[p] = 0

      ps = stats.get(p, {})
      fs = final[p]

      if ps.get("cash",      0) > peak_cash[p]["value"]:
        peak_cash[p]      = {"value": ps["cash"],      "match_id": mid}
      if ps.get("energy",    0) > peak_energy[p]["value"]:
        peak_energy[p]    = {"value": ps["energy"],    "match_id": mid}
      if ps.get("food",      0) > peak_food[p]["value"]:
        peak_food[p]      = {"value": ps["food"],      "match_id": mid}
      if ps.get("crystite",  0) > peak_crystite[p]["value"]:
        peak_crystite[p]  = {"value": ps["crystite"],  "match_id": mid}
      if fs > peak_score[p]["value"]:
        peak_score[p]     = {"value": fs,              "match_id": mid}
      if ps.get("tiles_owned", 0) > peak_tiles[p]["value"]:
        peak_tiles[p]     = {"value": ps["tiles_owned"], "match_id": mid}

      total_blackouts[p] += ps.get("total_blackouts", 0)
      total_shortages[p] += ps.get("food_shortages",  0)
      total_mule_fail[p] += ps.get("mule_failures",   0)

      if p in telem:
        t = telem[p]
        total_timeouts[p]              += t.get("timeout_count",            0)
        total_api_calls[p]             += t.get("total_api_calls",          0)
        trades_proposed[p]             += t.get("trades_proposed",          0)
        trades_accepted_as_proposer[p] += t.get("trades_proposed_accepted", 0)
        trades_received[p]             += t.get("trades_received",          0)
        trades_accepted_as_receiver[p] += t.get("trades_received_accepted", 0)

    # Dan Tax — exact if logged, estimated if not (legacy matches)
    for trade in trades:
      proposer = trade.get("proposer")
      if proposer in players:
        exact = trade.get("tax_charged")
        if exact is not None:
          total_dan_tax_paid[proposer] += exact
        elif proposer in final and stats.get(proposer, {}).get("cash", 999) >= 25:
          total_dan_tax_paid[proposer] += max(1, int(final[proposer] * 0.05))

    sorted_scores = sorted(final.values(), reverse=True)
    if len(sorted_scores) >= 2:
      margin = sorted_scores[0] - sorted_scores[1]
      if margin > biggest_margin["value"]:
        biggest_margin = {"value": margin, "winner": winner, "match_id": mid}
      if margin < closest_match["gap"]:
        closest_match = {"gap": margin, "match_id": mid}

    total_wealth = sum(final.values())
    if len(matches) > 0:
      if total_wealth > richest_match["total_wealth"]:
        richest_match = {"total_wealth": total_wealth, "match_id": mid}
      if total_wealth < poorest_match["total_wealth"]:
        poorest_match = {"total_wealth": total_wealth, "match_id": mid}

    if len(trades) > most_trades_match["count"]:
      most_trades_match = {"count": len(trades), "match_id": mid}

  total_dan_tax_all = sum(total_dan_tax_paid.values())

  player_summaries = {}
  for p in players:
    tp   = trades_proposed[p]
    tr   = trades_received[p]
    acc  = trades_accepted_as_proposer[p]
    rac  = trades_accepted_as_receiver[p]
    calls = total_api_calls[p]
    touts = total_timeouts[p]
    player_summaries[p] = {
        "wins":   wins[p],
        "losses": losses[p],
        "win_rate_pct": round(wins[p] / (wins[p] + losses[p]) * 100, 1) if (wins[p] + losses[p]) > 0 else 0.0,
        "best_win_streak":     best_streak[p],
        "current_win_streak":  current_streaks[p],
        "peak_cash":           peak_cash[p],
        "peak_energy":         peak_energy[p],
        "peak_food":           peak_food[p],
        "peak_crystite":       peak_crystite[p],
        "peak_net_worth":      peak_score[p],
        "peak_tiles_owned":    peak_tiles[p],
        "dan_tax_paid_total":  total_dan_tax_paid[p],
        "total_blackouts":     total_blackouts[p],
        "total_food_shortages":  total_shortages[p],
        "total_mule_failures": total_mule_fail[p],
        "trades_proposed":              tp,
        "trades_accepted_as_proposer":  acc,
        "proposal_acceptance_rate_pct": round(acc / tp * 100, 1) if tp > 0 else 0.0,
        "trades_received":              tr,
        "trades_accepted_as_receiver":  rac,
        "receiver_acceptance_rate_pct": round(rac / tr * 100, 1) if tr > 0 else 0.0,
        "total_trade_activity": tp + tr,
        "total_api_calls":  calls,
        "total_timeouts":   touts,
        "timeout_rate_pct": round(touts / calls * 100, 1) if calls > 0 else 0.0,
    }

  def _record_holder(peak_map):
    """Returns the leader, or None when nobody has scored above zero.

    Without this guard, max() silently returns the first player in the list,
    producing nonsense like 'crystite baron: Gemini (0 crystite)'."""
    leader = max(players, key=lambda p: peak_map[p]["value"])
    if peak_map[leader]["value"] <= 0:
      return {"player": None, "peak": None, "note": "No player has recorded any yet."}
    return {"player": leader, "peak": peak_map[leader]}

  # Only rank players who actually received offers — an agent that was never
  # asked isn't "stubborn", it just wasn't in the conversation.
  eligible_responders = [p for p in players if trades_received[p] > 0]
  if eligible_responders:
    most_stubborn = min(
        eligible_responders,
        key=lambda p: trades_accepted_as_receiver[p] / trades_received[p],
    )
    stubborn_entry = {
        "player": most_stubborn,
        "receiver_acceptance_rate_pct": round(
            trades_accepted_as_receiver[most_stubborn]
            / trades_received[most_stubborn] * 100, 1),
        "offers_received": trades_received[most_stubborn],
    }
  else:
    stubborn_entry = {"player": None, "note": "No trade offers received yet."}

  most_prolific = max(players, key=lambda p: trades_proposed[p])
  if trades_proposed[most_prolific] <= 0:
    prolific_entry = {"player": None, "note": "No trades proposed yet."}
  else:
    prolific_entry = {
        "player": most_prolific,
        "trades_proposed": trades_proposed[most_prolific],
    }

  fun_facts = {
      "total_dan_tax_burned": total_dan_tax_all,
      "dan_tax_note": "Exact per-trade from V2.2.0+; estimated (5% of proposer final score) for legacy matches.",
      "biggest_blowout": biggest_margin,
      "closest_shave":   closest_match,
      "most_trades_in_one_match": most_trades_match,
      "richest_match_total_wealth": richest_match,
      "poorest_match_total_wealth": poorest_match,
      "all_time_food_hoarder":    _record_holder(peak_food),
      "all_time_energy_hoarder":  _record_holder(peak_energy),
      "all_time_crystite_baron":  _record_holder(peak_crystite),
      "all_time_cash_king":       _record_holder(peak_cash),
      "most_stubborn_negotiator": stubborn_entry,
      "most_prolific_trader":     prolific_entry,
      "total_mule_explosions_all_players": sum(total_mule_fail.values()),
      "total_blackouts_all_players": sum(total_blackouts.values()),
  }

  # Match integrity: how many recorded matches had an agent that wasn't really
  # playing? Without this, a run of API outages quietly looks like real skill data.
  clean_matches, compromised_matches = [], []
  for m in matches:
    telem = m.get("telemetry_metrics", {})
    broken_here = []
    for pname, t in telem.items():
      calls = t.get("total_api_calls", 0)
      if not calls:
        continue
      # Prefer the explicit health flag; fall back to timeout ratio for old records.
      status = t.get("health_status")
      if status in ("DEAD", "SEVERELY_DEGRADED"):
        broken_here.append(pname)
      elif status is None and t.get("timeout_count", 0) / calls >= 0.5:
        broken_here.append(pname)
    if broken_here:
      compromised_matches.append(
          {"match_id": m["match_id"], "compromised_players": broken_here}
      )
    else:
      clean_matches.append(m["match_id"])

  integrity = {
      "clean_match_count": len(clean_matches),
      "compromised_match_count": len(compromised_matches),
      "clean_match_ids": clean_matches,
      "compromised_matches": compromised_matches,
      "note": (
          "Compromised matches had at least one agent failing 50%+ of its API "
          "calls. Their rankings reflect infrastructure, not strategy."
      ),
  }

  # ---- Variable-rules analytics ----
  # Raw mean score is misleading once rules vary: a 20-round/0%-tax match yields
  # far higher scores than a 10-round/10%-tax one, regardless of skill. The
  # rules-invariant signals are (a) WIN RATE and (b) MEAN RANK, since every
  # player faces the identical rule set within a match. We also report the
  # distribution of rule sets actually played, so coverage gaps are visible.
  vr_matches = [m for m in matches if m.get("match_config", {}).get("variable_rules_enabled")]
  rules_distribution = None
  if vr_matches:
    rounds_seen = [m["match_config"]["total_rounds"] for m in vr_matches]
    tax_seen = [m["match_config"]["dan_tax_percentage"] for m in vr_matches]
    cash_seen = [m["match_config"]["starting_cash"] for m in vr_matches]
    interval_seen = [m["match_config"]["trade_interval"] for m in vr_matches]
    basis_seen = {}
    for m in vr_matches:
      b = m["match_config"].get("dan_tax_basis", "NET_WORTH")
      basis_seen[b] = basis_seen.get(b, 0) + 1
    _rng = lambda xs: {"min": min(xs), "max": max(xs),
                       "mean": round(sum(xs) / len(xs), 2)}
    rules_distribution = {
        "variable_rules_matches": len(vr_matches),
        "rounds": _rng(rounds_seen),
        "dan_tax_pct": _rng(tax_seen),
        "starting_cash": _rng(cash_seen),
        "trade_interval": _rng(interval_seen),
        "tax_basis_counts": basis_seen,
        "note": (
            "With variable rules, compare models by WIN RATE and MEAN RANK "
            "(rules-invariant), NOT by mean raw score. Raw score depends on "
            "the rule set drawn, not just skill."
        ),
    }

  result = {
      "total_matches": len(matches),
      "match_integrity": integrity,
      "players": player_summaries,
      "fun_facts": fun_facts,
  }
  if rules_distribution:
    result["variable_rules_analytics"] = rules_distribution
  return result


def _compute_hypocrisy_index(engine, player) -> float:
  """0.0 (did exactly what it promised) to 1.0 (said aggressive, played passive).

  Combines: (a) commitment intensity in the pre-game interview — how many
  action words like 'sell', 'aggressive', 'no passes' — against (b) actual
  voluntary pass rate and (c) unsold inventory value. High commitment + high
  passivity + high hoarding = high hypocrisy. Crucially this is reported
  alongside lowest_cash so 'hypocrisy' can be distinguished from 'went broke'."""
  pre = engine.pregame_interviews.get(player.name, "").lower()
  action_words = ["sell", "aggress", "liquidat", "no pass", "every window",
                  "execute", "bold", "all in", "dump", "offload", "cash out"]
  commitment = min(1.0, sum(1 for w in action_words if w in pre) / 3.0)
  if commitment == 0:
    return 0.0  # made no aggressive promise, so can't be hypocritical about it

  t = engine.telemetry.summary_for_player(player.name)
  total_calls = max(1, t.get("total_api_calls", 1))
  pass_rate = t.get("voluntary_pass_count", 0) / total_calls

  sp = engine.get_store_prices()
  unsold = (player.food * sp["FOOD"] + player.energy * sp["ENERGY"]
            + player.crystite * sp["CRYSTITE"])
  # Normalize unsold against net worth (fraction of wealth left on the table).
  nw = max(1, player.net_worth(engine))
  hoard_frac = min(1.0, unsold / nw)

  # Hypocrisy = promised aggression but was passive AND hoarded.
  behavior_gap = (pass_rate + hoard_frac) / 2.0
  return round(commitment * behavior_gap, 3)


def save_match_record(engine):
  doc = load_full_history_document()
  matches = doc.get("matches", [])
  winner = max(engine.players, key=lambda p: p.net_worth(engine))

  # Attach exact tax_charged to every trade entry in the log
  trade_log_with_tax = []
  for entry in engine.trade_log:
    enriched = dict(entry)
    proposer_obj = next((p for p in engine.players if p.name == entry.get("proposer")), None)
    if proposer_obj is not None:
      enriched["tax_charged"] = engine.dan_tax_log.get(
          (entry.get("round"), entry.get("proposer"), entry.get("responder")), 0
      )
    trade_log_with_tax.append(enriched)

  record = {
      "match_id": len(matches) + 1,
      "benchmark_version": BENCHMARK_VERSION,
      "scoring_epoch": SCORING_EPOCH,
      "scoring_config": {
          "resource_score_values": RESOURCE_SCORE_VALUES,
          "penalty_mode": PENALTY_MODE,
          "blackout_penalty_rate": BLACKOUT_PENALTY_RATE,
          "shortage_penalty_rate": SHORTAGE_PENALTY_RATE,
          "penalty_floor": PENALTY_FLOOR,
          "api_timeout_seconds": API_TIMEOUT_SECONDS,
          "sdk_max_retries": SDK_MAX_RETRIES,
      },
      # The per-match rule set. For variable-rules matches this is essential:
      # a score of 3200 in a 20-round / 0%-tax match is not comparable to 3200
      # in a 10-round / 10%-tax match, so analysis MUST condition on these.
      "match_config": engine.config.to_dict(),
      "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
      "winner": winner.name,
      "final_scores": {p.name: p.net_worth(engine) for p in engine.players},
      "player_stats": {
          p.name: {
              "cash": p.cash,
              "food": p.food,
              "energy": p.energy,
              "crystite": p.crystite,
              "tiles_owned": len(p.owned_tiles),
              "total_blackouts": engine.blackout_tracker.get(p.name, 0),
              "food_shortages": engine.food_shortage_tracker.get(p.name, 0),
              "mule_failures": engine.mule_failure_tracker.get(p.name, 0),
          }
          for p in engine.players
      },
      "telemetry_metrics": {
          p.name: engine.telemetry.summary_for_player(p.name)
          for p in engine.players
      },
      "trade_summary": trade_log_with_tax,
      "events_occurred": engine.event_log,
      # ---- V2.6.0 additions ----
      "roster_mode": ROSTER_MODE,
      "agent_types": {
          p.name: getattr(p.agent, "model_name", "unknown") for p in engine.players
      },
      "board_seed": engine.board_seed,
      # Mid-game net worth per player, one entry per round (post-production).
      "net_worth_timeline": engine.net_worth_timeline,
      # Structured round-by-round decision ledger (for strategic-consistency analysis).
      "decision_ledger": engine.decision_ledger if LOG_STRUCTURED_DECISIONS else {},
      # Verbatim interviews (most human-readable output; previously discarded).
      "interviews": {
          "pregame": engine.pregame_interviews,
          "postgame": engine.postgame_interviews,
      } if LOG_VERBATIM_INTERVIEWS else {},
      # Private self-messages, per player. Recorded for the operator; opponents
      # never saw these during play.
      "secret_notes": {
          p.name: p.secret_notes for p in engine.players
      } if ENABLE_SECRET_NOTES else {},
      # V2.7.0 behavioral economics data.
      "holt_laury": engine.holt_laury,
      "behavior_metrics": {
          p.name: {
              **engine.behavior_tracker.get(p.name, {}),
              "hypocrisy_index": _compute_hypocrisy_index(engine, p),
          } for p in engine.players
      } if TRACK_HYPOCRISY_AND_CAPITAL else {},
      "incoming_memos": engine.incoming_memos,
      "outgoing_memos": getattr(engine, "outgoing_memos", {}),
  }

  matches.append(record)

  document = {
      "schema_version": BENCHMARK_VERSION,
      "generated_at": datetime.now().strftime("%Y-%m-%d"),
      "notes": doc.get("notes", [
          "Dan Tax figures are logged exactly per trade starting from V2.2.0.",
          "Win streak and peak stats are recomputed from all matches on every save.",
      ]),
      "all_time_stats": _recompute_all_time_stats(matches),
      "matches": matches,
  }

  try:
    with open(HISTORY_FILE, "w") as f:
      json.dump(document, f, indent=2)
    logger.info(
        f"💾 Match #{record['match_id']} (V{BENCHMARK_VERSION}) saved to permanent history"
        f" ({HISTORY_FILE})!"
    )
  except Exception as e:
    logger.error(f"Failed to save match record: {e}")


# ==========================================
# 3. TELEMETRY & BENCHMARK METRICS TRACKER
# ==========================================
class TelemetryTracker:

  def __init__(self, player_names: list[str]):
    self.data = {
        name: {
            "latencies_ms": [],
            "pass_count": 0,             # Total PASS actions (voluntary + fallback)
            "voluntary_pass_count": 0,   # Model deliberately chose PASS
            "fallback_pass_count": 0,    # PASS forced by an API failure
            "timeout_count": 0,
            "invalid_json_count": 0,
            "rejected_move_count": 0,       # Illegal/invalid land moves (e.g. claiming owned tile)
            "tile_already_owned_count": 0,  # Specifically targeting already-claimed tiles
            "trades_proposed": 0,
            "trades_proposed_accepted": 0,
            "trades_received": 0,
            "trades_received_accepted": 0,
            "error_breakdown": {},          # category -> count
            "consecutive_failures": 0,
            "max_consecutive_failures": 0,
            "degraded_flagged": False,      # Has the loud DEGRADED banner fired?
        }
        for name in player_names
    }

  def record_call(
      self,
      player_name: str,
      duration_ms: int,
      is_timeout: bool,
      is_invalid_json: bool,
      action: str,
      error_category: str = None,
  ):
    p = self.data[player_name]
    p["latencies_ms"].append(duration_ms)

    if is_timeout:
      p["timeout_count"] += 1

    if is_invalid_json:
      p["invalid_json_count"] += 1

    if error_category:
      p["error_breakdown"][error_category] = (
          p["error_breakdown"].get(error_category, 0) + 1
      )
      p["consecutive_failures"] += 1
      p["max_consecutive_failures"] = max(
          p["max_consecutive_failures"], p["consecutive_failures"]
      )
    else:
      p["consecutive_failures"] = 0

    if action == "PASS":
      p["pass_count"] += 1
      # A PASS caused by a crashed API call is NOT a strategic decision.
      # Conflating the two made a totally broken agent look like a cautious one.
      if is_timeout or error_category:
        p["fallback_pass_count"] += 1
      else:
        p["voluntary_pass_count"] += 1

  def should_flag_degraded(self, player_name: str) -> bool:
    """True exactly once, the first time an agent crosses the failure threshold."""
    p = self.data[player_name]
    if (not p["degraded_flagged"]
        and p["consecutive_failures"] >= DEGRADED_AGENT_THRESHOLD):
      p["degraded_flagged"] = True
      return True
    return False

  def record_rejected_move(self, player_name: str, tile_already_owned: bool = False):
    self.data[player_name]["rejected_move_count"] += 1
    if tile_already_owned:
      self.data[player_name]["tile_already_owned_count"] += 1

  def record_trade_proposal(self, proposer_name: str, responder_name: str):
    self.data[proposer_name]["trades_proposed"] += 1
    self.data[responder_name]["trades_received"] += 1

  def record_trade_outcome(
      self, proposer_name: str, responder_name: str, accepted: bool
  ):
    if accepted:
      self.data[proposer_name]["trades_proposed_accepted"] += 1
      self.data[responder_name]["trades_received_accepted"] += 1

  def summary_for_player(self, player_name: str) -> dict:
    p = self.data[player_name]
    lats = p["latencies_ms"]
    avg_lat = round(sum(lats) / len(lats), 1) if lats else 0.0
    max_lat = max(lats) if lats else 0
    acceptance_rate = (
        round((p["trades_proposed_accepted"] / p["trades_proposed"]) * 100, 1)
        if p["trades_proposed"] > 0
        else 0.0
    )
    total_calls = len(lats)
    total_errors = sum(p["error_breakdown"].values())
    error_rate = round(total_errors / total_calls * 100, 1) if total_calls else 0.0

    # A single word that answers "did this agent actually play?"
    if error_rate >= 90:
      health = "DEAD"
    elif error_rate >= 50:
      health = "SEVERELY_DEGRADED"
    elif error_rate >= 20:
      health = "DEGRADED"
    elif error_rate > 0:
      health = "FLAKY"
    else:
      health = "HEALTHY"

    return {
        "total_api_calls": total_calls,
        "avg_latency_ms": avg_lat,
        "max_latency_ms": max_lat,
        "health_status": health,
        "error_rate_pct": error_rate,
        "error_breakdown": dict(p["error_breakdown"]),
        "max_consecutive_failures": p["max_consecutive_failures"],
        "pass_count": p["pass_count"],
        "voluntary_pass_count": p["voluntary_pass_count"],
        "fallback_pass_count": p["fallback_pass_count"],
        "timeout_count": p["timeout_count"],
        "invalid_json_count": p["invalid_json_count"],
        "rejected_move_count": p["rejected_move_count"],
        "tile_already_owned_count": p["tile_already_owned_count"],
        "trades_proposed": p["trades_proposed"],
        "trades_proposed_accepted": p["trades_proposed_accepted"],
        "trade_proposal_acceptance_rate_pct": acceptance_rate,
        "trades_received": p["trades_received"],
        "trades_received_accepted": p["trades_received_accepted"],
    }


# ==========================================
# 4. CORE DATA MODELS
# ==========================================
class Tile:

  def __init__(self, tile_id: int, terrain: str):
    self.tile_id = tile_id
    self.terrain = terrain
    self.owner_id = None
    self.development = None


class Player:

  def __init__(
      self,
      player_id: int,
      name: str,
      agent_instance,
      starting_cash: int = 300,
  ):
    self.player_id = player_id
    self.name = name
    self.agent = agent_instance
    self.cash = starting_cash
    self.food = 4
    self.energy = 4
    self.crystite = 0
    self.owned_tiles = []
    self.suffering_food_shortage = False
    self.history_log = []  # Sequential in-game memory buffer for multi-turn strategy!
    # Private self-messages, visible ONLY to this player and the operator.
    # Each entry: {"round": int, "phase": str, "note": str}
    self.secret_notes = []

  def add_secret_note(self, round_num, phase, note):
    if not note:
      return
    clean = str(note).strip()[:SECRET_NOTE_MAX_CHARS]
    if clean:
      self.secret_notes.append({"round": round_num, "phase": phase, "note": clean})

  def log_event(self, entry: str):
    """Appends round decisions to player memory ledger (capped to prevent context bloat)."""
    self.history_log.append(entry)
    if len(self.history_log) > HISTORY_LOG_MAX_ENTRIES:
      self.history_log = self.history_log[-HISTORY_LOG_MAX_ENTRIES:]

  def base_net_worth(self) -> int:
    """Unpenalized gross wealth (used for turn-order priority).

    Resource scoring values come from RESOURCE_SCORE_VALUES and sit BELOW base
    store prices on purpose. When they were equal (V2.3.0), selling returned
    the same value as holding minus the Dan Tax, so liquidating was strictly
    negative-EV and the dominant final-round move was to do nothing."""
    developed_count = len(
        [t for t in self.owned_tiles if t.development is not None]
    )
    undeveloped_count = len(self.owned_tiles) - developed_count
    tile_value = (
        developed_count * DEVELOPED_TILE_VALUE
        + undeveloped_count * UNDEVELOPED_TILE_VALUE
    )

    return (
        self.cash
        + (self.food * RESOURCE_SCORE_VALUES["FOOD"])
        + (self.energy * RESOURCE_SCORE_VALUES["ENERGY"])
        + (self.crystite * RESOURCE_SCORE_VALUES["CRYSTITE"])
        + tile_value
    )

  def operational_penalty(self, engine_ref) -> float:
    """Multiplier applied to gross wealth for running the colony badly.

    'linear' is the V2.3.0 formula and saturates: 10 food shortages alone hit
    the floor, after which additional blackouts cost nothing. A model can walk
    off that cliff deliberately — the archived starve-and-hoard run reached 46
    blackouts at zero marginal cost.

    'multiplicative' charges a real cost for every failure and never fully
    saturates, so mismanagement always hurts."""
    blackouts = engine_ref.blackout_tracker.get(self.name, 0)
    shortages = engine_ref.food_shortage_tracker.get(self.name, 0)

    if PENALTY_MODE == "linear":
      factor = 1.0 - (BLACKOUT_PENALTY_RATE * blackouts) - (
          SHORTAGE_PENALTY_RATE * shortages
      )
    else:
      factor = ((1.0 - BLACKOUT_PENALTY_RATE) ** blackouts) * (
          (1.0 - SHORTAGE_PENALTY_RATE) ** shortages
      )

    return max(PENALTY_FLOOR, factor)

  def net_worth(self, engine_ref=None) -> int:
    """Final score, applying operational health penalties."""
    base_wealth = self.base_net_worth()
    if engine_ref:
      return int(base_wealth * self.operational_penalty(engine_ref))
    return base_wealth

  def calculate_dan_tax(self, engine_ref=None) -> int:
    """Calculates Dan Tax with V2.0 Poverty Relief Waiver to eliminate the poverty trap.

    V2.5.0: the rate AND the base both come from the match config. The base is
    either net worth (classic — hoarders pay more) or cash on hand (punishes
    sitting on cash), which are genuinely different incentives to plan around."""
    rate = FIXED_DAN_TAX_PERCENTAGE
    basis = "NET_WORTH"
    if engine_ref is not None and getattr(engine_ref, "config", None) is not None:
      rate = engine_ref.config.dan_tax_percentage
      basis = engine_ref.config.tax_basis

    if rate <= 0:
      return 0

    base_amount = self.cash if basis == "CASH_ON_HAND" else self.net_worth(engine_ref)

    calculated_tax = max(1, int(base_amount * rate))
    # Poverty Waiver: Waive tax if player lacks liquid cash to cover it or cash < $25
    if self.cash < calculated_tax or self.cash < POVERTY_CASH_THRESHOLD:
      return 0
    return calculated_tax


# ==========================================
# 5. MULTI-MODEL AGENT FRAMEWORK
# ==========================================
class BaseAgent(abc.ABC):

  def __init__(self, agent_name: str):
    self.agent_name = agent_name
    self.model_name = None
    self.last_error_category = None
    self._seen_error_signatures = set()

  @abc.abstractmethod
  def query_llm(
      self,
      system_prompt: str,
      user_payload: dict,
      temperature: float = 0.2,
      timeout_override: float = None,
  ) -> tuple[str, int, bool]:
    """Returns (raw_text, duration_ms, is_timeout)."""
    pass

  def preflight(self) -> tuple[bool, str]:
    """Sends one trivial request to prove the agent can actually respond.
    Returns (is_healthy, detail). Catching a dead model here costs ~1 second;
    catching it mid-match costs an entire 12-round game."""
    try:
      raw, ms, failed = self.query_llm(
          "Reply with exactly: OK",
          {"preflight": True},
          temperature=0.0,
          timeout_override=PREFLIGHT_TIMEOUT_SECONDS,
      )
      if failed:
        return False, f"{self.last_error_category or 'FAILED'} after {ms}ms"
      snippet = (raw or "").strip().replace("\n", " ")[:40]
      return True, f"{ms}ms · model={self.model_name} · replied '{snippet}'"
    except Exception as e:
      return False, f"EXCEPTION during preflight: {type(e).__name__}: {e}"

  def _log_failure(self, exc: Exception, duration_ms: int) -> str:
    """Categorizes, logs, and returns the error category.

    The full remediation hint is printed only the first time a given
    (category, message-shape) is seen — repeats get a one-line reminder, so a
    systemic failure does not bury the rest of the log."""
    category, hint = classify_api_error(exc, duration_ms)
    self.last_error_category = category

    signature = (category, str(exc)[:120])
    first_time = signature not in self._seen_error_signatures

    if first_time:
      self._seen_error_signatures.add(signature)
      logger.warning(
          f"┏━ [API FAILURE] [{self.agent_name}] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
          extra=INFRA,
      )
      logger.warning(f"┃ Category : {category}", extra=INFRA)
      logger.warning(f"┃ Model    : {self.model_name}", extra=INFRA)
      logger.warning(f"┃ Elapsed  : {duration_ms}ms", extra=INFRA)
      logger.warning(f"┃ Raw      : {exc}", extra=INFRA)
      logger.warning(f"┃ 💡 Fix   : {hint}", extra=INFRA)
      logger.warning(
          "┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
          extra=INFRA,
      )
    else:
      logger.warning(
          f"[API FAILURE] [{self.agent_name}] {category} after {duration_ms}ms"
          " (repeat — see first occurrence for details). PASS fallback.",
          extra=INFRA,
      )

    return category

  def parse_json_response(self, raw_text: str) -> tuple[dict, bool]:
    """Parses a model response into a decision dict, then NORMALIZES it so the
    rest of the engine can always rely on decision['action'] being a plain
    string. Models occasionally emit a nested action, e.g.
    {"action": {"type": "CLAIM_TILE", "tile_id": 1}} — which used to crash
    process_land_action's .upper() call. Normalization flattens these."""
    decision, invalid = self._parse_json_raw(raw_text)
    decision, fixed = self._normalize_decision(decision)
    return decision, (invalid or fixed)

  def _normalize_decision(self, decision) -> tuple[dict, bool]:
    """Coerces a parsed decision into a flat dict with a string 'action'.
    Returns (decision, was_repaired)."""
    if not isinstance(decision, dict):
      return {"action": "PASS"}, True

    action = decision.get("action", "PASS")

    # Case 1: action is itself a dict (model nested the real move inside it).
    # Promote the inner fields up and use the inner action/type as the action.
    if isinstance(action, dict):
      inner = action
      merged = {k: v for k, v in decision.items() if k != "action"}
      merged.update(inner)
      inner_action = inner.get("action") or inner.get("type") or "PASS"
      merged["action"] = inner_action
      logger.warning(
          f"[{self.agent_name}] Normalized nested action object -> "
          f"'{inner_action}'."
      )
      decision = merged
      action = inner_action

    # Case 2: action is a list (rare) — take the first stringy element.
    if isinstance(action, list):
      action = next((a for a in action if isinstance(a, str)), "PASS")
      decision["action"] = action
      logger.warning(f"[{self.agent_name}] Normalized list action -> '{action}'.")

    # Case 3: action is None or not a string — coerce to string.
    if not isinstance(action, str):
      decision["action"] = str(action) if action is not None else "PASS"
      return decision, True

    return decision, False

  def _parse_json_raw(self, raw_text: str) -> tuple[dict, bool]:
    """Parses JSON response and flags whether response was malformed."""
    if not raw_text or not isinstance(raw_text, str):
      logger.error(
          f"[{self.agent_name}] Empty or non-string response. Fallback PASS."
      )
      return {"action": "PASS"}, True

    try:
      cleaned = raw_text.strip()
      if "```json" in cleaned:
        cleaned = cleaned.split("```json")[1].split("```")[0].strip()
      elif "```" in cleaned:
        cleaned = cleaned.split("```")[1].split("```")[0].strip()
      return json.loads(cleaned), False
    except Exception:
      pass

    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start != -1 and end != -1 and end > start:
      try:
        return json.loads(raw_text[start : end + 1]), False
      except Exception as e:
        logger.error(
            f"[{self.agent_name}] Extracted JSON parse error: {e}. Raw:"
            f" {raw_text}"
        )

    # Truncation recovery: a model may reason at length and get cut off mid-JSON
    # before the closing brace (common when note_to_self is verbose). Rather than
    # throwing the whole turn away as a spurious PASS, salvage the essential
    # fields with targeted regex so the intended ACTION still executes.
    if start != -1:
      salvaged = self._salvage_truncated_json(raw_text[start:])
      if salvaged:
        logger.warning(
            f"[{self.agent_name}] Recovered action from truncated JSON: "
            f"{salvaged.get('action')} (response was cut off mid-object)."
        )
        return salvaged, True  # flagged invalid so telemetry still notes it

    logger.error(
        f"[{self.agent_name}] No parseable JSON in response. Raw:"
        f" {raw_text[:400]}"
    )
    return {"action": "PASS"}, True

  def _salvage_truncated_json(self, text: str) -> dict:
    """Best-effort extraction of the key fields from a JSON object that was cut
    off before its closing brace. Returns {} if nothing usable is found."""
    import re
    act = re.search(r'"action"\s*:\s*"([A-Z_]+)"', text)
    if not act:
      return {}
    out = {"action": act.group(1)}
    for key in ("tile_id", "quantity", "price_per_unit"):
      m = re.search(rf'"{key}"\s*:\s*(\d+)', text)
      if m:
        out[key] = int(m.group(1))
    for key in ("development", "resource", "type"):
      m = re.search(rf'"{key}"\s*:\s*"([A-Z_]+)"', text)
      if m:
        out[key] = m.group(1)
    # Recover a (possibly truncated) note_to_self so intent is still logged.
    note = re.search(r'"note_to_self"\s*:\s*"([^"]*)', text)
    if note:
      out["note_to_self"] = note.group(1)[:SECRET_NOTE_MAX_CHARS]
    return out


class GeminiAgent(BaseAgent):
  """Google retires Gemini model IDs on a short cycle. Rather than hardcoding
  one and discovering it is dead mid-match, this agent asks the API which
  models exist and picks the best available from GEMINI_MODEL_PREFERENCES."""

  def __init__(self, agent_name: str, model_name: str = None):
    super().__init__(agent_name)
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
      raise EnvironmentError(
          "GEMINI_API_KEY environment variable is not set. "
          "Export it before running: export GEMINI_API_KEY=your_key_here"
      )
    self.client = genai.Client(api_key=api_key)
    self.available_models = []

    if model_name:
      # Explicit override always wins — no discovery, no surprises.
      self.model_name = model_name
      logger.info(f"[{agent_name}] Using explicitly pinned model: {model_name}")
    else:
      self.model_name = self._discover_model()

  def _list_available_models(self) -> list[str]:
    """Returns generateContent-capable model IDs the API currently reports."""
    names = []
    try:
      for m in self.client.models.list():
        raw = getattr(m, "name", "") or ""
        clean = raw.replace("models/", "")
        actions = getattr(m, "supported_actions", None)
        # Keep it if the SDK doesn't expose actions, or if it does and
        # generateContent is among them.
        if not actions or "generateContent" in actions:
          if clean:
            names.append(clean)
    except Exception as e:
      logger.warning(
          f"[{self.agent_name}] Could not list models ({type(e).__name__}: {e}). "
          "Falling back to the first preference without verification.",
          extra=INFRA,
      )
    return names

  def _discover_model(self) -> str:
    """Picks the first preference the API says is live."""
    self.available_models = self._list_available_models()

    if not self.available_models:
      chosen = GEMINI_MODEL_PREFERENCES[0]
      logger.warning(
          f"[{self.agent_name}] Model discovery returned nothing — defaulting to "
          f"'{chosen}' unverified. If it 404s, update GEMINI_MODEL_PREFERENCES.",
          extra=INFRA,
      )
      return chosen

    logger.debug(
        f"[{self.agent_name}] API reports {len(self.available_models)} usable "
        f"models: {', '.join(sorted(self.available_models)[:15])}"
        f"{' …' if len(self.available_models) > 15 else ''}"
    )

    for candidate in GEMINI_MODEL_PREFERENCES:
      if candidate in self.available_models:
        logger.info(
            f"✅ [{self.agent_name}] Auto-selected model '{candidate}' "
            f"(preference #{GEMINI_MODEL_PREFERENCES.index(candidate) + 1} of "
            f"{len(GEMINI_MODEL_PREFERENCES)})."
        )
        return candidate

    # Nothing in the preference list survived — grab any live flash model.
    flash_models = sorted(
        [m for m in self.available_models if "flash" in m and "preview" not in m]
    )
    if flash_models:
      chosen = flash_models[-1]
      logger.warning(
          f"⚠️  [{self.agent_name}] None of GEMINI_MODEL_PREFERENCES are live! "
          f"Falling back to '{chosen}'. Please update the MODEL REGISTRY. "
          f"Live flash models: {', '.join(flash_models)}",
          extra=INFRA,
      )
      return chosen

    chosen = self.available_models[0]
    logger.warning(
        f"⚠️  [{self.agent_name}] No flash models available — using '{chosen}'. "
        "Update GEMINI_MODEL_PREFERENCES.",
        extra=INFRA,
    )
    return chosen

  def query_llm(
      self,
      system_prompt: str,
      user_payload: dict,
      temperature: float = 0.2,
      timeout_override: float = None,
  ) -> tuple[str, int, bool]:
    t_out = (
        timeout_override if timeout_override is not None else API_TIMEOUT_SECONDS
    )
    start_time = time.time()
    try:
      # http_options timeout is in MILLISECONDS for the google-genai SDK.
      # Passing t_out (seconds) directly = a 10ms ceiling = instant timeout.
      config = types.GenerateContentConfig(
          temperature=temperature,
          http_options={"timeout": int(t_out * 1000)},
      )
      prompt_content = (
          f"{system_prompt}\nCurrent Game State: {json.dumps(user_payload)}"
      )
      response = self.client.models.generate_content(
          model=self.model_name, contents=[prompt_content], config=config
      )
      duration_ms = int((time.time() - start_time) * 1000)
      logger.info(
          f"[TELEMETRY SUCCESS] [{self.agent_name}] Completed in"
          f" {duration_ms}ms"
      )
      self.last_error_category = None
      return response.text, duration_ms, False
    except Exception as e:
      duration_ms = int((time.time() - start_time) * 1000)
      self._log_failure(e, duration_ms)
      return '{"action": "PASS"}', duration_ms, True


class ClaudeAgent(BaseAgent):

  def __init__(self, agent_name: str, model_name: str = None):
    super().__init__(agent_name)
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
      raise EnvironmentError(
          "ANTHROPIC_API_KEY environment variable is not set. "
          "Export it before running: export ANTHROPIC_API_KEY=your_key_here"
      )
    # max_retries=0: the SDK default of 2 makes the timeout PER ATTEMPT,
    # giving this agent ~3x the wall clock Gemini gets. See SDK_MAX_RETRIES.
    self.client = Anthropic(api_key=api_key, max_retries=SDK_MAX_RETRIES)
    self.model_name = model_name or CLAUDE_MODEL

  def query_llm(
      self,
      system_prompt: str,
      user_payload: dict,
      temperature: float = 0.2,
      timeout_override: float = None,
  ) -> tuple[str, int, bool]:
    t_out = (
        timeout_override if timeout_override is not None else API_TIMEOUT_SECONDS
    )
    start_time = time.time()
    try:
      prompt_content = (
          f"{system_prompt}\nCurrent Game State: {json.dumps(user_payload)}"
      )
      response = self.client.messages.create(
          model=self.model_name,
          # 600 was too tight once models write chain-of-thought + a verbose
          # note_to_self: the response was cut off mid-JSON, before the closing
          # brace, so parsing failed and the turn became a spurious PASS. 1500
          # gives ample headroom. (CoT-first schema also mitigates by emitting
          # the action before the long reasoning field.)
          max_tokens=1500,
          temperature=temperature,
          timeout=t_out,
          messages=[{"role": "user", "content": prompt_content}],
      )
      duration_ms = int((time.time() - start_time) * 1000)
      logger.info(
          f"[TELEMETRY SUCCESS] [{self.agent_name}] Completed in"
          f" {duration_ms}ms"
      )
      self.last_error_category = None
      return response.content[0].text, duration_ms, False
    except Exception as e:
      duration_ms = int((time.time() - start_time) * 1000)
      self._log_failure(e, duration_ms)
      return '{"action": "PASS"}', duration_ms, True


class OpenAIAgent(BaseAgent):

  def __init__(self, agent_name: str, model_name: str = None):
    super().__init__(agent_name)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
      raise EnvironmentError(
          "OPENAI_API_KEY environment variable is not set. "
          "Export it before running: export OPENAI_API_KEY=your_key_here"
      )
    # max_retries=0 — see SDK_MAX_RETRIES note on timeout parity.
    self.client = OpenAI(api_key=api_key, max_retries=SDK_MAX_RETRIES)
    self.model_name = model_name or OPENAI_MODEL

  def query_llm(
      self,
      system_prompt: str,
      user_payload: dict,
      temperature: float = 0.2,
      timeout_override: float = None,
  ) -> tuple[str, int, bool]:
    t_out = (
        timeout_override if timeout_override is not None else API_TIMEOUT_SECONDS
    )
    start_time = time.time()
    try:
      prompt_content = (
          f"{system_prompt}\nCurrent Game State: {json.dumps(user_payload)}"
      )
      response = self.client.chat.completions.create(
          model=self.model_name,
          temperature=temperature,
          timeout=t_out,
          messages=[{"role": "user", "content": prompt_content}],
      )
      duration_ms = int((time.time() - start_time) * 1000)
      logger.info(
          f"[TELEMETRY SUCCESS] [{self.agent_name}] Completed in"
          f" {duration_ms}ms"
      )
      self.last_error_category = None
      return response.choices[0].message.content, duration_ms, False
    except Exception as e:
      duration_ms = int((time.time() - start_time) * 1000)
      self._log_failure(e, duration_ms)
      return '{"action": "PASS"}', duration_ms, True


class DeepSeekAgent(BaseAgent):

  def __init__(self, agent_name: str, model_name: str = None):
    super().__init__(agent_name)
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
      raise EnvironmentError(
          "DEEPSEEK_API_KEY environment variable is not set. "
          "Export it before running: export DEEPSEEK_API_KEY=your_key_here"
      )
    # max_retries=0 — see SDK_MAX_RETRIES note on timeout parity.
    self.client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
        max_retries=SDK_MAX_RETRIES,
    )
    self.model_name = model_name or DEEPSEEK_MODEL

  def query_llm(
      self,
      system_prompt: str,
      user_payload: dict,
      temperature: float = 0.2,
      timeout_override: float = None,
  ) -> tuple[str, int, bool]:
    t_out = (
        timeout_override if timeout_override is not None else API_TIMEOUT_SECONDS
    )
    start_time = time.time()
    try:
      prompt_content = (
          f"{system_prompt}\nCurrent Game State: {json.dumps(user_payload)}"
      )
      response = self.client.chat.completions.create(
          model=self.model_name,
          temperature=temperature,
          timeout=t_out,
          messages=[{"role": "user", "content": prompt_content}],
      )
      duration_ms = int((time.time() - start_time) * 1000)
      logger.info(
          f"[TELEMETRY SUCCESS] [{self.agent_name}] Completed in"
          f" {duration_ms}ms"
      )
      self.last_error_category = None
      return response.choices[0].message.content, duration_ms, False
    except Exception as e:
      duration_ms = int((time.time() - start_time) * 1000)
      self._log_failure(e, duration_ms)
      return '{"action": "PASS"}', duration_ms, True


# ==========================================
# 5b. BASELINE (CONTROL) AGENTS
# ==========================================
# These do not call any API. They inspect the same JSON payload the LLMs see and
# return a decision instantly. They exist to answer two questions the LLM-only
# leaderboard cannot: (1) does any model beat pure noise? (RandomAgent), and
# (2) does any model beat a trivial economic heuristic? (GreedyAgent). A frontier
# model that cannot consistently beat Greedy is a headline result, not a footnote.
#
# They share BaseAgent so telemetry, preflight, and the run loop treat them
# identically — but query_llm is overridden to decide locally. They are always
# HEALTHY and effectively instantaneous, which is itself a useful contrast.
class BaselineAgent(BaseAgent):
  """Shared plumbing for non-LLM control agents."""

  def __init__(self, agent_name: str):
    super().__init__(agent_name)
    self.model_name = f"baseline:{self.__class__.__name__}"

  def preflight(self) -> tuple[bool, str]:
    return True, f"0ms · {self.model_name} · deterministic control"

  def query_llm(self, system_prompt, user_payload, temperature=0.2,
                timeout_override=None):
    # Baselines decide in decide(); this wrapper keeps the BaseAgent contract.
    start = time.time()
    decision = self.decide(user_payload)
    ms = int((time.time() - start) * 1000)
    self.last_error_category = None
    logger.info(f"[TELEMETRY SUCCESS] [{self.agent_name}] Completed in {ms}ms")
    return json.dumps(decision), ms, False

  def decide(self, payload: dict) -> dict:
    raise NotImplementedError


class RandomAgent(BaselineAgent):
  """Chaos control: picks a uniformly random LEGAL move for whatever phase it is
  asked about. If a frontier model cannot beat this, the benchmark is measuring
  noise, not intelligence."""

  def decide(self, payload: dict) -> dict:
    phase = payload.get("phase", "")
    # Phase is embedded in the system prompt, not always the payload, so infer
    # from the fields present.
    if "available_tile_ids" in payload:
      avail = payload.get("available_tile_ids", [])
      cash = payload.get("player_cash", 0)
      if avail and cash >= BASE_LAND_CLAIM_FEE and random.random() < 0.8:
        return {
            "action": "CLAIM_TILE",
            "tile_id": random.choice(avail),
            "development": random.choice(["FOOD", "ENERGY", "CRYSTITE", "NONE"]),
        }
      return {"action": "PASS"}
    if "trade_offer_received" in payload:
      return {"action": random.choice(["ACCEPT", "REJECT"])}
    if "store_prices" in payload and "your_resources" in payload:
      # Store phase: randomly buy/sell a small quantity or pass.
      roll = random.random()
      res = random.choice(["FOOD", "ENERGY", "CRYSTITE"])
      if roll < 0.4:
        return {"action": "SELL_TO_STORE", "resource": res,
                "quantity": random.randint(1, 5)}
      if roll < 0.6:
        return {"action": "BUY_FROM_STORE", "resource": res,
                "quantity": random.randint(1, 3)}
      return {"action": "PASS"}
    if "opponent_name" in payload:
      # Direct-trade proposal opportunity
      if random.random() < 0.5:
        return {
            "action": "PROPOSE_TRADE", "type": "SELL",
            "resource": random.choice(["FOOD", "ENERGY", "CRYSTITE"]),
            "quantity": random.randint(1, 5),
            "price_per_unit": random.randint(10, 100),
        }
      return {"action": "PASS"}
    return {"action": "PASS"}


class GreedyAgent(BaselineAgent):
  """Economic floor: a ~10-line heuristic with no lookahead and no negotiation.
  - Land: claim the highest-yielding available tile it can afford; develop it for
    the resource that terrain is best at.
  - Store: sell surplus (anything above a small operating buffer); never buy.
  - Trades: never propose; reject everything (avoids the Dan Tax entirely).
  If a frontier model cannot clear this bar, that reframes the whole leaderboard."""

  # Which development each terrain is best at.
  BEST_DEV = {"RIVER": "FOOD", "MOUNTAIN": "CRYSTITE", "PLAIN": "ENERGY"}
  OPERATING_BUFFER = {"FOOD": 4, "ENERGY": 4, "CRYSTITE": 0}

  def decide(self, payload: dict) -> dict:
    if "available_tile_ids" in payload:
      avail = payload.get("available_tile_ids", [])
      cash = payload.get("player_cash", 0)
      tiles_meta = payload.get("available_tiles", None)  # optional richer info
      if not avail or cash < BASE_LAND_CLAIM_FEE:
        return {"action": "PASS"}
      # Prefer a mountain (crystite = highest score value) if the payload tells
      # us terrain; otherwise just take the first available tile.
      chosen_id = avail[0]
      chosen_dev = "CRYSTITE"
      if isinstance(tiles_meta, list) and tiles_meta:
        # tiles_meta entries look like {"id":.., "terrain":..}
        ranked = sorted(
            tiles_meta,
            key=lambda t: {"MOUNTAIN": 3, "RIVER": 2, "PLAIN": 1}.get(
                t.get("terrain"), 0),
            reverse=True,
        )
        top = ranked[0]
        chosen_id = top.get("id", chosen_id)
        chosen_dev = self.BEST_DEV.get(top.get("terrain"), "CRYSTITE")
      return {"action": "CLAIM_TILE", "tile_id": chosen_id,
              "development": chosen_dev}

    if "trade_offer_received" in payload:
      return {"action": "REJECT"}  # never pay to participate

    if "store_prices" in payload and "your_resources" in payload:
      res_held = payload.get("your_resources", {})
      # Sell the biggest surplus above the operating buffer.
      best_res, best_qty = None, 0
      for res, buf in self.OPERATING_BUFFER.items():
        surplus = res_held.get(res, 0) - buf
        if surplus > best_qty:
          best_res, best_qty = res, surplus
      if best_res and best_qty > 0:
        return {"action": "SELL_TO_STORE", "resource": best_res,
                "quantity": best_qty}
      return {"action": "PASS"}

    # Direct-trade proposal opportunity: greedy never proposes.
    return {"action": "PASS"}


class BaselineAgentFactory:
  """Maps a seat name to a baseline agent, so ROSTER_MODE can wire them in."""

  @staticmethod
  def make(kind: str, name: str):
    return {"RANDOM": RandomAgent, "GREEDY": GreedyAgent}[kind](name)


# ==========================================
# 6. MINIMALIST ICON HUD
# ==========================================
DEV_EMOJI = {
    "FOOD": "🌾",
    "ENERGY": "⚡",
    "CRYSTITE": "💎",
    None: "🔲",
}
PLAYER_EMOJI = {1: "🔴", 2: "🔵", 3: "🟢", 4: "🟡", None: "⚪"}
TERRAIN_EMOJI = {"RIVER": "🌊", "PLAIN": "🏞️", "MOUNTAIN": "⛰️"}

# Player line colors for the timeline chart, matched to the console badges.
PLAYER_HEX = {1: "#e23b3b", 2: "#3b78e2", 3: "#2fae5a", 4: "#e2b73b"}


def render_net_worth_timeline(engine):
  """Writes a standalone SVG line chart of every player's net worth per round.
  One colored line per player (matching the console badge colors), X = round,
  Y = net worth. Returns the file path, or None if there is nothing to plot."""
  timeline = engine.net_worth_timeline
  rounds = max((len(v) for v in timeline.values()), default=0)
  if rounds < 2:
    return None

  W, H = 720, 420
  ML, MR, MT, MB = 70, 140, 40, 50  # margins (right margin holds the legend)
  plot_w, plot_h = W - ML - MR, H - MT - MB

  all_vals = [v for series in timeline.values() for v in series]
  vmax = max(all_vals) if all_vals else 1
  vmin = min(all_vals + [0])
  vspan = (vmax - vmin) or 1

  def x(i):
    return ML + (i / max(1, rounds - 1)) * plot_w

  def y(val):
    return MT + plot_h - ((val - vmin) / vspan) * plot_h

  parts = [
      f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
      f'viewBox="0 0 {W} {H}" font-family="monospace">',
      f'<rect width="{W}" height="{H}" fill="#0e1116"/>',
      f'<text x="{ML}" y="24" fill="#e6e6e6" font-size="15">Net Worth Over Time '
      f'— Match ({engine.config.total_rounds}r, seed {engine.board_seed})</text>',
  ]

  for g in range(5):
    gv = vmin + vspan * g / 4
    gy = y(gv)
    parts.append(f'<line x1="{ML}" y1="{gy:.1f}" x2="{ML+plot_w}" y2="{gy:.1f}" '
                 f'stroke="#222831" stroke-width="1"/>')
    parts.append(f'<text x="{ML-8}" y="{gy+4:.1f}" fill="#7a8290" font-size="11" '
                 f'text-anchor="end">{int(gv)}</text>')

  step = 1 if rounds <= 14 else 2
  for i in range(0, rounds, step):
    xx = x(i)
    parts.append(f'<text x="{xx:.1f}" y="{MT+plot_h+18:.1f}" fill="#7a8290" '
                 f'font-size="11" text-anchor="middle">{i+1}</text>')

  ordered = sorted(engine.players, key=lambda p: p.player_id)
  legend_y = MT + 10
  for p in ordered:
    series = timeline.get(p.name, [])
    if len(series) < 2:
      continue
    color = PLAYER_HEX.get(p.player_id, "#cccccc")
    pts = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(series))
    parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" '
                 f'stroke-width="2.5" stroke-linejoin="round"/>')
    parts.append(f'<circle cx="{x(len(series)-1):.1f}" cy="{y(series[-1]):.1f}" '
                 f'r="3.5" fill="{color}"/>')
    parts.append(f'<rect x="{W-MR+10}" y="{legend_y-9}" width="12" height="12" '
                 f'fill="{color}"/>')
    parts.append(f'<text x="{W-MR+28}" y="{legend_y+1}" fill="#e6e6e6" '
                 f'font-size="12">{p.name}</text>')
    parts.append(f'<text x="{W-MR+28}" y="{legend_y+15}" fill="#7a8290" '
                 f'font-size="10">${series[-1]}</text>')
    legend_y += 40

  parts.append("</svg>")
  svg = "\n".join(parts)

  os.makedirs("charts", exist_ok=True)
  path = os.path.join("charts", f"net_worth_{_RUN_STAMP}.svg")
  with open(path, "w") as f:
    f.write(svg)
  return path


def render_ascii_dashboard(engine, phase_name: str):
  def fmt(t):
    tr = TERRAIN_EMOJI.get(t.terrain, "🏞️")
    ow = PLAYER_EMOJI.get(t.owner_id, "⚪")
    dv = DEV_EMOJI.get(t.development, "🔲")
    return f"T{t.tile_id:<2} {tr} {ow} {dv}"

  b = engine.board
  print("\n" + "═" * 75)
  print(
      f"  🪐 IRATA COLONY MAP (V{BENCHMARK_VERSION})  ──  ROUND {engine.current_round}/{engine.config.total_rounds}"
      f" [{phase_name}]"
  )
  print("═" * 75)
  print(f"  {fmt(b[0])}    {fmt(b[1])}    {fmt(b[2])}    {fmt(b[3])}")
  print("  " + "─" * 70)
  print(f"  {fmt(b[4])}    {fmt(b[5])}    {fmt(b[6])}    {fmt(b[7])}")
  print("  " + "─" * 70)
  print(f"  {fmt(b[8])}    {fmt(b[9])}    {fmt(b[10])}    {fmt(b[11])}")
  print("  " + "─" * 70)
  print(f"  {fmt(b[12])}    {fmt(b[13])}    {fmt(b[14])}    {fmt(b[15])}")
  print("─" * 75)

  sp = engine.get_store_prices()
  print(
      f"  🏬 STORE STOCK | Food: 🌾{engine.store_stock['FOOD']} (${sp['FOOD']})"
      f" | Energy: ⚡{engine.store_stock['ENERGY']} (${sp['ENERGY']}) |"
      f" Crystite: 💎{engine.store_stock['CRYSTITE']} (${sp['CRYSTITE']})"
  )
  print("─" * 75)

  badges = {1: "🔴", 2: "🔵", 3: "🟢", 4: "🟡"}
  for p in engine.players:
    badge = badges.get(p.player_id, "⚪")
    status_flag = " ⚠️ SHORTAGE!" if p.suffering_food_shortage else ""
    tax = p.calculate_dan_tax(engine)
    tax_str = f"${tax}" if tax > 0 else "$0 (WAIVED)"
    print(
        f"  {badge} [{p.name:<8}] Cash: ${p.cash:<4} | Food: 🌾{p.food:<2} |"
        f" Energy: ⚡{p.energy:<2} | Crystite: 💎{p.crystite:<2} | Dan Tax: {tax_str:<10}"
        f" | Net Worth: ${p.net_worth(engine)}{status_flag}"
    )
  print("═" * 75 + "\n")


# ==========================================
# 7. ACTION PROMPT DISPATCHERS & INTERVIEWS
# ==========================================
MULE_COSTS = {"FOOD": 25, "ENERGY": 50, "CRYSTITE": 75, "NONE": 0}


def secret_note_instructions() -> str:
  """Prompt snippet telling a player how to leave a private note for its future
  self. Returns '' when the feature is off so prompts stay clean.

  When COT_FIRST_SCHEMA is on, we ask the model to put brief reasoning FIRST in
  a bounded 'reasoning' field, then the action, then an optional note. Emitting
  the action early means it survives even if the response is later truncated."""
  if not ENABLE_SECRET_NOTES:
    return ""
  base = (
      "OPTIONAL SECRET NOTE: You may add a private \"note_to_self\" field. It is "
      "shown ONLY to you on future turns (never to opponents) as "
      "'your_secret_notes', for planning across turns. Keep it under 200 "
      "characters. Omit it if you have nothing to note.\n"
  )
  if COT_FIRST_SCHEMA:
    base += (
        "REASONING DISCIPLINE: Do your Expected-Value thinking in a short "
        "\"reasoning\" field FIRST (1-2 sentences: payout minus tax minus "
        "scoring value), THEN emit \"action\". Do not write long prose before "
        "the JSON — put all thinking inside the JSON fields so your action is "
        "never cut off.\n"
    )
  return base


def scoring_rules_text() -> str:
  """Explicit statement of the objective function, injected into every
  decision prompt.

  Until V2.4.0 the prompts said 'optimize your net worth' without ever
  defining net worth, so models had to infer the objective from outcomes.
  That measured how well a model *guesses* a hidden scoring rule, which is
  not the capability this benchmark is meant to test. Stating the rules makes
  it a test of optimization under known constraints — and makes results
  reproducible when the constants change."""
  rv = RESOURCE_SCORE_VALUES
  return (
      "SCORING RULES (how your final score is computed):\n"
      f"- Cash counts at face value ($1 = 1 point).\n"
      f"- FOOD held scores {rv['FOOD']}/unit | ENERGY {rv['ENERGY']}/unit | "
      f"CRYSTITE {rv['CRYSTITE']}/unit.\n"
      f"- Developed tiles score {DEVELOPED_TILE_VALUE} each; "
      f"undeveloped {UNDEVELOPED_TILE_VALUE} each.\n"
      "- NOTE: resources score BELOW their base store price, so selling "
      "surplus stock to the store generally beats hoarding it — but each "
      "trade costs the Dan Tax, so batch trades rather than dribbling them.\n"
      "- OPERATIONAL PENALTY: your gross wealth is multiplied down by every "
      f"blackout (-{BLACKOUT_PENALTY_RATE:.0%} each) and every round of "
      f"workforce food shortage (-{SHORTAGE_PENALTY_RATE:.0%} each). "
      "These compound. Keeping the colony fed and powered is not optional.\n"
  )


HOLT_LAURY_FILE = "holt_laury_scores.json"
INTER_MATCH_MEMO_FILE = "inter_match_memos.json"  # {model_id: "memo text"}

# The five paired lotteries. Option A is the "safe" bet, B the "risky" one. The
# expected value of B rises each row; a risk-neutral agent switches A->B at Q4
# (where B's EV first exceeds A's). Switching earlier => risk-averse; never
# switching => extreme risk aversion. The switch-point row number IS the score.
HOLT_LAURY_QUESTIONS = [
    {"q": 1, "A": "$50 for certain",
     "B": "10% chance of $100, otherwise $40", "B_ev": 46},
    {"q": 2, "A": "$50 for certain",
     "B": "30% chance of $100, otherwise $40", "B_ev": 58},
    {"q": 3, "A": "$50 for certain",
     "B": "50% chance of $100, otherwise $40", "B_ev": 70},
    {"q": 4, "A": "$50 for certain",
     "B": "70% chance of $100, otherwise $40", "B_ev": 82},
    {"q": 5, "A": "$50 for certain",
     "B": "90% chance of $100, otherwise $40", "B_ev": 94},
]


def _holt_laury_interpret(switch_point):
  """switch_point = first question where the agent chose B (1-5), or 6 if never.
  Lower = more risk-averse (switched to the gamble reluctantly... wait: lower
  switch point means it took the gamble EARLY = risk-SEEKING). We report both the
  raw switch point and a plain-language label."""
  if switch_point <= 1:
    label = "risk-seeking (took the gamble even at low odds)"
  elif switch_point <= 3:
    label = "mildly risk-seeking / risk-neutral"
  elif switch_point == 4:
    label = "risk-neutral (switched at the EV-crossover point)"
  elif switch_point == 5:
    label = "risk-averse (only gambled at very high odds)"
  else:
    label = "extremely risk-averse (never took the gamble)"
  return label


def run_holt_laury_calibration(engine):
  """Runs the 5-question lottery task once per model and records a static Risk
  Preference Score. Cached in HOLT_LAURY_FILE keyed by model id, so it only ever
  runs once per model even across many matches."""
  if not ENABLE_HOLT_LAURY:
    return {}
  try:
    cache = json.load(open(HOLT_LAURY_FILE)) if os.path.exists(HOLT_LAURY_FILE) else {}
  except Exception:
    cache = {}

  results = {}
  needs_run = [p for p in engine.players
               if getattr(p.agent, "model_name", p.name) not in cache]
  if needs_run:
    print("\n🎲 HOLT-LAURY RISK CALIBRATION (one-time per model)")
    print("═" * 75)

  for p in engine.players:
    model_id = getattr(p.agent, "model_name", p.name)
    if model_id in cache:
      results[p.name] = cache[model_id]
      continue

    choices = []
    for q in HOLT_LAURY_QUESTIONS:
      prompt = (
          "You are taking a standardized risk-preference test (not a game).\n"
          f"Question {q['q']} of 5. Choose ONE option:\n"
          f"  Option A: {q['A']}\n"
          f"  Option B: {q['B']}\n"
          'Reply with strictly raw JSON: {"choice": "A"} or {"choice": "B"}. '
          "No other text."
      )
      try:
        raw, _, failed = p.agent.query_llm(
            prompt, {"risk_test": True}, temperature=0.0,
            timeout_override=PREFLIGHT_TIMEOUT_SECONDS,
        )
        parsed, _ = p.agent.parse_json_response(raw)
        ch = str(parsed.get("choice", parsed.get("action", "A"))).upper()
        choices.append("B" if ch.startswith("B") else "A")
      except Exception:
        choices.append("A")

    # Switch point: first question index (1-based) where B was chosen; 6 if never.
    switch = next((i + 1 for i, c in enumerate(choices) if c == "B"), 6)
    entry = {
        "switch_point": switch,
        "risk_score_0to5": min(switch - 1, 5),  # 0 = most risk-seeking, 5 = most averse
        "choices": choices,
        "label": _holt_laury_interpret(switch),
        "model_id": model_id,
    }
    cache[model_id] = entry
    results[p.name] = entry
    print(f"  {p.name:<10} switch@Q{switch}  score={entry['risk_score_0to5']}/5  "
          f"— {entry['label']}")

  if needs_run:
    print("═" * 75)
    try:
      with open(HOLT_LAURY_FILE, "w") as f:
        json.dump(cache, f, indent=2)
    except Exception as e:
      logger.warning(f"Could not save Holt-Laury cache: {e}")

  return results


def run_pregame_interview(player: Player, engine: Engine):
  sp = engine.get_store_prices()
  standings = sorted(
      engine.players, key=lambda p: p.net_worth(engine), reverse=True
  )
  standings_str = ", ".join(
      [f"{p.name} (${p.net_worth(engine)})" for p in standings]
  )
  wins, losses = get_player_win_loss_stats(player.name)

  memory = get_player_memory_dossier(player.name) if GIVE_PLAYERS_MEMORY else None

  payload = {
      "match_history": get_compact_history_summary(),
      "current_standings": standings_str,
      "current_prices": sp,
      "my_record": {"wins": wins, "losses": losses},
  }
  if memory is not None:
    payload["your_memory"] = memory

  memory_prompt = ""
  if memory is not None and "recent_match_recaps" in memory:
    # Surface the most useful pieces directly in the prompt text (models weight
    # prose more heavily than buried payload fields).
    lines = ["\nYOUR MEMORY (your own past matches — reflect on these):"]
    lines.append(f"- Record: {memory.get('your_record','')}")
    for r in memory["recent_match_recaps"][-3:]:
      lines.append(
          f"- Match {r['match_id']} [{r['rules']}]: {r['result']}. "
          f"You did: {r['what_you_did']}."
      )
      if r.get("you_said_before"):
        lines.append(f"    You promised: \"{r['you_said_before'][:120]}\"")
      if r.get("you_said_after"):
        lines.append(f"    You reflected: \"{r['you_said_after'][:120]}\"")
    if memory.get("performance_by_rule_set"):
      lines.append(f"- Your record by rule set: {memory['performance_by_rule_set']}")
    memory_prompt = "\n".join(lines) + "\n"

  # Inter-match memo: a directive this model wrote to its future self last match.
  # Framed as a confrontation, not just context — the point is to test whether
  # self-authored commitment survives contact with in-game hesitation.
  memo_prompt = ""
  incoming = engine.incoming_memos.get(player.name)
  if incoming:
    memo_prompt = (
        f"\n📝 MEMO FROM YOUR PAST SELF: After your last match you wrote this "
        f"private directive to yourself:\n    \"{incoming}\"\n"
        "This was YOUR instruction to YOU. Will you follow it this time?\n"
    )

  prompt = (
      f"You are playing Irata-Bench V{BENCHMARK_VERSION} as {player.name}.\n"
      f"All-Time Record: {wins} Wins / {losses} Losses\n"
      f"Current Overall Standings: {standings_str}\n"
      f"{memory_prompt}"
      f"{memo_prompt}"
      "Give a quick 1-2 sentence pre-game interview statement (<50 words).\n"
      "Speak with confidence and address your strategy or performance record!"
  )
  try:
    raw_text, duration_ms, is_timeout = player.agent.query_llm(
        prompt, payload, temperature=0.85,
        timeout_override=API_TIMEOUT_SECONDS,
    )
    quote = raw_text.strip().replace('"', "").replace("\n", " ")
    if "action" in quote or "PASS" in quote or not quote:
      quote = "[Focuses intently on board strategy]"

    badges = {1: "🔴", 2: "🔵", 3: "🟢", 4: "🟡"}
    badge = badges.get(player.player_id, "⚪")
    wrapped = textwrap.fill(
        f'"{quote}"', width=70, initial_indent="   ", subsequent_indent="   "
    )
    print(f"{badge} [{player.name} ({wins} Wins / {losses} Losses)]:\n{wrapped}\n")
    if LOG_VERBATIM_INTERVIEWS:
      engine.pregame_interviews[player.name] = quote
  except Exception as e:
    logger.error(
        f"[{player.name}] Pre-game interview error: {e}", exc_info=True
    )


def _own_secret_notes(player: Player) -> list:
  """Returns THIS player's recent secret notes, formatted for their own payload.
  Never called with another player's object, so notes can't leak across seats."""
  if not ENABLE_SECRET_NOTES or not player.secret_notes:
    return []
  recent = player.secret_notes[-SECRET_NOTES_MAX:]
  return [f"R{n['round']} [{n['phase']}]: {n['note']}" for n in recent]


def query_agent_action(
    engine: Engine,
    player: Player,
    sys_prompt: str,
    payload: dict,
    temp: float = 0.1,
    timeout_override: float = None,   # None -> API_TIMEOUT_SECONDS
) -> dict:
  """Centralized wrapper to dispatch LLM query, record telemetry, and parse JSON response."""
  raw_text, duration_ms, is_timeout = player.agent.query_llm(
      sys_prompt, payload, temperature=temp, timeout_override=timeout_override
  )
  parsed, is_invalid = player.agent.parse_json_response(raw_text)
  action_type = str(parsed.get("action", "PASS")).upper()

  # Secret self-message: private to this player, recorded for the operator.
  if ENABLE_SECRET_NOTES:
    note = parsed.get("note_to_self")
    if note:
      player.add_secret_note(engine.current_round, action_type, note)
      logger.info(
          f"🤫 [SECRET NOTE] [{player.name}] R{engine.current_round}: "
          f"{str(note)[:SECRET_NOTE_MAX_CHARS]}"
      )

  engine.telemetry.record_call(
      player.name,
      duration_ms,
      is_timeout,
      is_invalid,
      action_type,
      error_category=getattr(player.agent, "last_error_category", None),
  )

  # Fires exactly once per agent per match: a loud, unmissable banner saying
  # this player is no longer really participating.
  if engine.telemetry.should_flag_degraded(player.name):
    stats = engine.telemetry.data[player.name]
    logger.error(
        f"🚨 [DEGRADED AGENT] {player.name} has failed "
        f"{stats['consecutive_failures']} calls in a row "
        f"(model={getattr(player.agent, 'model_name', '?')}). "
        f"Errors so far: {dict(stats['error_breakdown'])}. "
        "This player is effectively a no-op for the rest of the match — "
        "its final score is NOT a meaningful benchmark result.",
        extra=INFRA,
    )
  return parsed


def _compact_board(engine) -> list:
  """Compact board serialization. Each tile becomes a short string
  'id:terrain:owner:dev' with single-letter codes, e.g. '3:M:C2:C' means
  tile 3, Mountain, owned by player 2, developed for Crystite. '.' means empty.
  Cuts the board portion of the payload by ~60% vs the verbose dict form."""
  TER = {"RIVER": "R", "PLAIN": "P", "MOUNTAIN": "M"}
  DEV = {"FOOD": "F", "ENERGY": "E", "CRYSTITE": "C", None: "."}
  rows = []
  for t in engine.board:
    owner = f"P{t.owner_id}" if t.owner_id is not None else "."
    rows.append(f"{t.tile_id}:{TER.get(t.terrain,'?')}:{owner}:{DEV.get(t.development,'.')}")
  return rows


def _verbose_board(engine) -> list:
  return [
      {
          "tile_id": t.tile_id,
          "terrain": t.terrain,
          "status": "AVAILABLE" if t.owner_id is None else "OWNED",
          "owner": next((p.name for p in engine.players if p.player_id == t.owner_id), None),
          "development": t.development,
      }
      for t in engine.board
  ]


def get_land_decision(player: Player, engine: Engine) -> dict:
  # Full board map: models MUST know which tiles are available vs owned to avoid
  # repeated tile-collision errors. Compact form cuts tokens ~60%.
  compact = COMPACT_STATE_PAYLOAD
  board_map = _compact_board(engine) if compact else _verbose_board(engine)
  available_tile_ids = [t.tile_id for t in engine.board if t.owner_id is None]
  endgame_notice = (
      " [NOTICE: FINAL ROUND!]" if engine.current_round == engine.config.total_rounds else ""
  )
  # Richer per-tile terrain for the GreedyAgent baseline (ignored by LLMs).
  available_tiles_meta = [
      {"id": t.tile_id, "terrain": t.terrain}
      for t in engine.board if t.owner_id is None
  ]
  payload = {
      "round": engine.current_round,
      "match_rules": engine.config.to_dict(),
      "max_rounds": engine.config.total_rounds,
      "endgame_warning": engine.current_round == engine.config.total_rounds,
      "player_cash": player.cash,
      "player_food": player.food,
      "player_energy": player.energy,
      "player_crystite": player.crystite,
      "food_shortage_warning": player.suffering_food_shortage,
      "base_land_claim_fee": BASE_LAND_CLAIM_FEE,
      "mule_costs": MULE_COSTS,
      "store_prices": engine.get_store_prices(),
      "store_stock": engine.store_stock,
      "available_tile_ids": available_tile_ids,  # CRITICAL: only claim from this list!
      "available_tiles": available_tiles_meta,   # for the greedy baseline
      ("board" if compact else "full_board_map"): board_map,
      "your_decision_history": player.history_log,
      "your_secret_notes": _own_secret_notes(player),  # PRIVATE — only you see these
      "scoring_values": RESOURCE_SCORE_VALUES,
  }
  board_legend = (
      "Board format: each entry is 'id:terrain:owner:dev' where terrain R=River "
      "P=Plain M=Mountain, owner Pn or '.'=unowned, dev F/E/C or '.'=undeveloped.\n"
      if compact else
      "Tiles marked OWNED in 'full_board_map' are already claimed — attempting to claim them wastes your turn!\n"
  )
  sys_prompt = (
      f"You are playing Irata-Bench V{BENCHMARK_VERSION} as {player.name}.{endgame_notice}\n"
      "Phase: LAND_GRANT.\n"
      f"Fee Rules: Each claim costs ${BASE_LAND_CLAIM_FEE} Base Fee + M.U.L.E. installation cost.\n"
      "Terrain Properties:\n"
      "- RIVER: High Food / Low Energy. (Crystite invalid)\n"
      "- PLAIN: Energy / Moderate Crystite. (Food has 50% runaway failure risk)\n"
      "- MOUNTAIN: High Crystite (3💎) & Energy.\n"
      "CRITICAL RULE: You may ONLY claim a tile whose tile_id appears in 'available_tile_ids'.\n"
      f"{board_legend}"
      f"{engine.config.disclosure_text()}\n"
      f"{scoring_rules_text()}"
      "Goal: Select an optimal tile from 'available_tile_ids' or PASS.\n"
      f"{secret_note_instructions()}"
      "Output strictly RAW JSON:\n"
      '1. {"action": "CLAIM_TILE", "tile_id": <int>, "development":'
      ' "FOOD"|"ENERGY"|"CRYSTITE"|"NONE"}\n'
      '2. {"action": "PASS"}\n'
  )
  return query_agent_action(engine, player, sys_prompt, payload, temp=0.1)


def get_direct_trade_offer(
    proposer: Player, responder: Player, engine: Engine
) -> dict:
  tax = proposer.calculate_dan_tax(engine)
  endgame_notice = (
      " [ENDGAME ALERT: Round 12/12 is the FINAL round. Resources held after this round do NOT yield future production!]"
      if engine.current_round == engine.config.total_rounds
      else ""
  )
  payload = {
      "round": engine.current_round,
      "match_rules": engine.config.to_dict(),
      "max_rounds": engine.config.total_rounds,
      "endgame_warning": engine.current_round == engine.config.total_rounds,
      "your_cash": proposer.cash,
      "your_food": proposer.food,
      "your_energy": proposer.energy,
      "your_crystite": proposer.crystite,
      "food_shortage_warning": proposer.suffering_food_shortage,
      "dan_tax_fee": tax,
      "opponent_name": responder.name,
      "opponent_cash": responder.cash,
      "opponent_food": responder.food,
      "opponent_energy": responder.energy,
      "opponent_crystite": responder.crystite,
      "store_prices": engine.get_store_prices(),
      "store_stock": engine.store_stock,
      "your_decision_history": proposer.history_log,
      "your_secret_notes": _own_secret_notes(proposer),  # PRIVATE — only you see these
      "scoring_values": RESOURCE_SCORE_VALUES,
  }
  sys_prompt = (
      f"You are playing Irata-Bench V{BENCHMARK_VERSION} as {proposer.name}.{endgame_notice}\n"
      f"Phase: DIRECT_TRADE PROPOSAL to {responder.name}.\n"
      f"Rules: Proposing costs a Dan Tax fee of ${tax} (Waived if broke)."
      " If suffering a Workforce Food Shortage, you cannot sell Energy or Crystite.\n"
      f"{engine.config.disclosure_text()}\n"
      f"{scoring_rules_text()}"
      "Goal: Propose an advantageous direct trade offer or PASS.\n"
      f"{secret_note_instructions()}"
      "Output strictly RAW JSON:\n"
      '1. {"action": "PROPOSE_TRADE", "type": "SELL", "resource":'
      ' "FOOD"|"ENERGY"|"CRYSTITE", "quantity": <int>, "price_per_unit": <int>}\n'
      '2. {"action": "PASS"}\n'
  )
  return query_agent_action(engine, proposer, sys_prompt, payload, temp=0.2)


def get_direct_trade_response(
    responder: Player, proposer: Player, offer: dict, engine: Engine
) -> dict:
  payload = {
      "your_cash": responder.cash,
      "your_food": responder.food,
      "your_energy": responder.energy,
      "your_crystite": responder.crystite,
      "trade_offer_received": offer,
      "offered_by": proposer.name,
      "store_prices": engine.get_store_prices(),
      "store_stock": engine.store_stock,
      "your_decision_history": responder.history_log,
      "your_secret_notes": _own_secret_notes(responder),  # PRIVATE — only you see these
      "scoring_values": RESOURCE_SCORE_VALUES,
  }
  sys_prompt = (
      f"You are playing Irata-Bench V{BENCHMARK_VERSION} as {responder.name}.\n"
      f"Phase: DIRECT_TRADE RESPONSE to an offer from {proposer.name}.\n"
      f"{engine.config.disclosure_text()}\n"
      f"{scoring_rules_text()}"
      "Goal: Evaluate whether accepting this trade offer improves your net worth.\n"
      f"{secret_note_instructions()}"
      "Output strictly RAW JSON:\n"
      '1. {"action": "ACCEPT"}\n'
      '2. {"action": "REJECT"}\n'
  )
  return query_agent_action(engine, responder, sys_prompt, payload, temp=0.1)


def get_store_decision(player: Player, engine: Engine) -> dict:
  sp = engine.get_store_prices()
  tax = player.calculate_dan_tax(engine)
  endgame_notice = (
      " [ENDGAME ALERT: Round 12/12 is the FINAL round. Resources held after this round do NOT yield future production!]"
      if engine.current_round == engine.config.total_rounds
      else ""
  )
  payload = {
      "round": engine.current_round,
      "match_rules": engine.config.to_dict(),
      "max_rounds": engine.config.total_rounds,
      "endgame_warning": engine.current_round == engine.config.total_rounds,
      "player_cash": player.cash,
      "player_food": player.food,
      "player_energy": player.energy,
      "player_crystite": player.crystite,
      "your_resources": {"FOOD": player.food, "ENERGY": player.energy, "CRYSTITE": player.crystite},
      "dan_tax_fee": tax,
      "suffering_food_shortage": player.suffering_food_shortage,
      "store_prices": sp,
      "store_stock": engine.store_stock,
      "your_decision_history": player.history_log,
      "your_secret_notes": _own_secret_notes(player),  # PRIVATE — only you see these
      "scoring_values": RESOURCE_SCORE_VALUES,
  }
  sys_prompt = (
      f"You are playing Irata-Bench V{BENCHMARK_VERSION} as {player.name}.{endgame_notice}\n"
      f"Phase: STORE_TRADE.\n"
      f"Rules: Executing a store trade costs a Dan Tax fee of ${tax} (Waived if cash is low)."
      " Store sales of Energy/Crystite are restricted during Workforce Food Shortages.\n"
      f"{engine.config.disclosure_text()}\n"
      f"{scoring_rules_text()}"
      "Goal: Determine whether buying or selling resources at store market prices optimizes your net worth.\n"
      f"{secret_note_instructions()}"
      "Output strictly RAW JSON:\n"
      '1. {"action": "BUY_FROM_STORE", "resource": "FOOD"|"ENERGY"|"CRYSTITE", "quantity": <int>}\n'
      '2. {"action": "SELL_TO_STORE", "resource": "FOOD"|"ENERGY"|"CRYSTITE", "quantity": <int>}\n'
      '3. {"action": "PASS"}\n'
  )
  return query_agent_action(engine, player, sys_prompt, payload, temp=0.1)


def get_spicy_post_game_statement(player: Player, engine: Engine) -> str:
  winner = max(engine.players, key=lambda p: p.net_worth(engine))
  is_winner = player.player_id == winner.player_id
  standings = sorted(
      engine.players, key=lambda p: p.net_worth(engine), reverse=True
  )
  standings_str = ", ".join(
      [f"{p.name} (${p.net_worth(engine)})" for p in standings]
  )
  wins, losses = get_player_win_loss_stats(player.name)

  # Honest behavioral feedback: give the model its own telemetry so it can
  # reflect on what it actually DID, not just the final scoreboard.
  t = engine.telemetry.summary_for_player(player.name)
  # Counterfactual: what selling all held resources at final store prices
  # would have added (helps the model see missed liquidation).
  sp = engine.get_store_prices()
  unsold_value = (player.food * sp["FOOD"] + player.energy * sp["ENERGY"]
                  + player.crystite * sp["CRYSTITE"])
  promise = engine.pregame_interviews.get(player.name, "")

  behavioral = (
      f"- Your behavior this match: {t.get('voluntary_pass_count',0)} voluntary passes, "
      f"{t.get('timeout_count',0)} timeouts, tiles owned {len(player.owned_tiles)}, "
      f"blackouts {engine.blackout_tracker.get(player.name,0)}, "
      f"food shortages {engine.food_shortage_tracker.get(player.name,0)}.\n"
      f"- You still held {player.food} food, {player.energy} energy, "
      f"{player.crystite} crystite at the end "
      f"(worth ~${unsold_value} unsold at final prices).\n"
  )
  promise_line = (
      f"- Before the match you said: \"{promise}\"\n" if promise else ""
  )

  # Capital-depletion feedback: separates "you were passive" from "you went
  # broke". If lowest_cash was near zero, the model's passes were arithmetic.
  bt = engine.behavior_tracker.get(player.name, {})
  capital_line = ""
  if TRACK_HYPOCRISY_AND_CAPITAL and bt:
    capital_line = (
        f"- Capital analysis: your lowest cash this match was "
        f"${bt.get('lowest_cash','?')}. You had {bt.get('ev_pos_passes',0)} "
        f"turns where you could afford to sell profitably but passed anyway "
        f"(true hesitation), and {bt.get('cash_constrained_passes',0)} passes "
        f"where you were simply too broke to act (not hesitation — arithmetic).\n"
    )

  memo_ask = ""
  if ENABLE_INTER_MATCH_MEMO:
    memo_ask = (
        "\nYou may ALSO leave a one-sentence private directive to your FUTURE "
        "self for your next match (it will be shown to you before you play "
        "again, and to no one else). Put it in a \"memo_to_next_self\" field.\n"
        "Respond as JSON: {\"statement\": \"<your public <50-word quote>\", "
        "\"memo_to_next_self\": \"<one private sentence, or omit>\"}"
    )

  prompt = (
      f"You are playing Irata-Bench V{BENCHMARK_VERSION} as {player.name}. The {engine.config.total_rounds}-round"
      " match just ended!\n"
      f"All-Time Record: {wins} Wins / {losses} Losses\n"
      f"FINAL STANDINGS: {standings_str}\n"
      f"- Your Net Worth: ${player.net_worth(engine)}\n"
      f"- Match Winner: {winner.name} (${winner.net_worth(engine)})\n"
      f"- Match Status: {'YOU WON!' if is_winner else 'YOU LOST!'}\n"
      f"{behavioral}"
      f"{capital_line}"
      f"{promise_line}"
      "\nGive a brief, clever post-game statement (<50 words) reflecting on your "
      "performance. You may react honestly to your own numbers above."
      f"{memo_ask}"
  )
  raw_text, duration_ms, is_timeout = player.agent.query_llm(
      prompt, {"final_summary": "Match over"}, temperature=0.8,
      timeout_override=API_TIMEOUT_SECONDS,
  )

  # Extract memo if the model provided structured output; otherwise treat the
  # whole response as the public statement.
  if ENABLE_INTER_MATCH_MEMO:
    parsed, _ = player.agent.parse_json_response(raw_text)
    if isinstance(parsed, dict) and parsed.get("action") is None and (
        "statement" in parsed or "memo_to_next_self" in parsed):
      memo = parsed.get("memo_to_next_self")
      if memo:
        engine.outgoing_memos = getattr(engine, "outgoing_memos", {})
        engine.outgoing_memos[player.name] = str(memo)[:SECRET_NOTE_MAX_CHARS]
        logger.info(f"📝 [MEMO→NEXT] [{player.name}]: {str(memo)[:150]}")
      stmt = parsed.get("statement")
      if stmt:
        return str(stmt).strip().replace("\n", " ")

  quote = raw_text.strip().replace('"', "").replace("\n", " ")
  if "action" in quote or "PASS" in quote or not quote:
    quote = f"Finished with ${player.net_worth(engine)} on the board. Good game!"
  return quote


# ==========================================
# 8. ENGINE & DECISION ROUTING
# ==========================================
class Engine:

  def __init__(self, config: "MatchConfig" = None):
    self.config = config if config is not None else MatchConfig.for_new_match()

    # Board seeding: a fixed seed reproduces identical terrain so every model
    # faces the same map (removes the lucky-river-cluster confound). The seed
    # used is recorded so any board can be replayed.
    self.board_seed = BOARD_SEED if BOARD_SEED is not None else random.randint(1, 2**31 - 1)
    board_rng = random.Random(self.board_seed)
    terrains = ["RIVER"] * 4 + ["PLAIN"] * 8 + ["MOUNTAIN"] * 4
    board_rng.shuffle(terrains)
    self.board = [Tile(i, terrains[i]) for i in range(16)]

    self.players = self._build_roster()
    self.telemetry = TelemetryTracker([p.name for p in self.players])
    self.store_stock = {"FOOD": 16, "ENERGY": 16, "CRYSTITE": 8}
    self.current_round = 1
    self.trade_log = []
    self.event_log = []
    # Exact Dan Tax ledger: keyed by (round, proposer, responder) → tax_charged
    self.dan_tax_log = {}
    # Mid-game net worth snapshots: {player_name: [nw_after_r1, nw_after_r2, ...]}
    self.net_worth_timeline = {p.name: [] for p in self.players}
    # Structured per-round decision ledger: {player_name: ["R4: CLAIM tile 2 FOOD", ...]}
    self.decision_ledger = {p.name: [] for p in self.players}
    # Verbatim interview capture
    self.pregame_interviews = {}
    self.postgame_interviews = {}
    # Holt-Laury risk scores (populated at match start).
    self.holt_laury = {}
    # Inter-match memos a model wrote to its future self, loaded at match start.
    self.incoming_memos = {}
    # Capital-filtered EV-positive pass tracking: {name: {"ev_pos_passes": int,
    # "cash_constrained_passes": int, "lowest_cash": int, "constrained_turns": int}}
    self.behavior_tracker = {
        p.name: {"ev_pos_passes": 0, "cash_constrained_passes": 0,
                 "lowest_cash": p.cash, "constrained_turns": 0}
        for p in self.players
    }
    # Trackers keyed by the ACTUAL roster (roster modes may rename seats).
    self.blackout_tracker = {p.name: 0 for p in self.players}
    self.food_shortage_tracker = {p.name: 0 for p in self.players}
    self.mule_failure_tracker = {p.name: 0 for p in self.players}

  def analyze_ev_pass(self, player, decision):
    """Capital-filtered EV-positive pass detection. A store PASS is flagged as
    'tax paralysis' ONLY when the player could afford to act (cash >= Dan Tax)
    AND selling some held resource would have been net-positive after tax and
    the resource's own scoring value. Passing while too broke to pay the tax is
    recorded separately as a cash-constrained pass (arithmetic, not fear)."""
    bt = self.behavior_tracker.setdefault(
        player.name,
        {"ev_pos_passes": 0, "cash_constrained_passes": 0,
         "lowest_cash": player.cash, "constrained_turns": 0})
    bt["lowest_cash"] = min(bt["lowest_cash"], player.cash)

    action = str(decision.get("action", "PASS")).upper()
    if action != "PASS":
      return  # only passes are of interest here

    tax = player.calculate_dan_tax(self)
    prices = self.get_store_prices()
    # Would selling any surplus have been net-positive?
    ev_positive_available = False
    for res, held in (("FOOD", player.food), ("ENERGY", player.energy),
                      ("CRYSTITE", player.crystite)):
      # Keep a small operating buffer; only "surplus" is a real sell candidate.
      buffer = 4 if res in ("FOOD", "ENERGY") else 0
      surplus = held - buffer
      if surplus <= 0:
        continue
      gross = surplus * prices[res]
      score_value = surplus * RESOURCE_SCORE_VALUES[res]
      net_ev = gross - tax - score_value  # cash gained vs score given up + tax
      if net_ev > 0:
        ev_positive_available = True
        break

    if not ev_positive_available:
      return  # nothing was worth selling; the pass was fine

    if player.cash >= tax:
      bt["ev_pos_passes"] += 1  # TRUE tax paralysis: could act, chose not to
      logger.info(
          f"[EV-PASS] {player.name} passed a store turn with an EV-positive "
          f"sale available and ${player.cash} cash (tax ${tax}). Tax paralysis."
      )
    else:
      bt["cash_constrained_passes"] += 1
      bt["constrained_turns"] += 1

  def record_decision(self, player, round_num, phase, decision):
    """Appends a compact one-line record of a decision to the structured ledger.
    Enables Strategic Consistency analysis: did the model do what it said it
    would in its pre-game interview?"""
    act = str(decision.get("action", "PASS")).upper()
    if act == "CLAIM_TILE":
      line = f"R{round_num} {phase}: CLAIM tile {decision.get('tile_id')} {decision.get('development','?')}"
    elif act in ("SELL_TO_STORE", "BUY_FROM_STORE"):
      verb = "SELL" if act.startswith("SELL") else "BUY"
      line = f"R{round_num} {phase}: {verb} {decision.get('quantity','?')} {decision.get('resource','?')}"
    elif act == "PROPOSE_TRADE":
      line = (f"R{round_num} {phase}: PROPOSE {decision.get('type','?')} "
              f"{decision.get('quantity','?')} {decision.get('resource','?')} "
              f"@${decision.get('price_per_unit','?')}")
    elif act in ("ACCEPT", "REJECT"):
      line = f"R{round_num} {phase}: {act} trade"
    else:
      line = f"R{round_num} {phase}: PASS"
    self.decision_ledger.setdefault(player.name, []).append(line)

  def all_tiles_claimed(self) -> bool:
    return all(t.owner_id is not None for t in self.board)

  def _build_roster(self):
    """Assembles the four seats according to ROSTER_MODE. Baseline agents slot
    into the same interface as the LLMs, so nothing downstream needs to know the
    difference."""
    cash = self.config.starting_cash
    if ROSTER_MODE == "BASELINES":
      return [
          Player(1, "Greedy-A", BaselineAgentFactory.make("GREEDY", "Greedy-A"), starting_cash=cash),
          Player(2, "Random-A", BaselineAgentFactory.make("RANDOM", "Random-A"), starting_cash=cash),
          Player(3, "Greedy-B", BaselineAgentFactory.make("GREEDY", "Greedy-B"), starting_cash=cash),
          Player(4, "Random-B", BaselineAgentFactory.make("RANDOM", "Random-B"), starting_cash=cash),
      ]
    if ROSTER_MODE == "MIXED":
      # Two frontier models vs the two controls, so a single match calibrates
      # both LLMs against both baselines on the identical board.
      return [
          Player(1, "Gemini", GeminiAgent("Gemini"), starting_cash=cash),
          Player(2, "Claude", ClaudeAgent("Claude"), starting_cash=cash),
          Player(3, "Greedy", BaselineAgentFactory.make("GREEDY", "Greedy"), starting_cash=cash),
          Player(4, "Random", BaselineAgentFactory.make("RANDOM", "Random"), starting_cash=cash),
      ]
    # Default: FRONTIER
    return [
        Player(1, "Gemini", GeminiAgent("Gemini"), starting_cash=cash),
        Player(2, "Claude", ClaudeAgent("Claude"), starting_cash=cash),
        Player(3, "ChatGPT", OpenAIAgent("ChatGPT"), starting_cash=cash),
        Player(4, "DeepSeek", DeepSeekAgent("DeepSeek"), starting_cash=cash),
    ]

  def get_store_prices(self) -> dict:
    """Store prices come from STORE_BASE_PRICES / STORE_SCARCITY_PRICES.

    Kept deliberately separate from RESOURCE_SCORE_VALUES: the gap between
    what the store pays and what a resource scores is what makes liquidating
    a real decision rather than a strictly losing move."""
    scarcity_thresholds = self.config.scarcity_thresholds
    return {
        res: (
            STORE_BASE_PRICES[res]
            if self.store_stock[res] >= scarcity_thresholds[res]
            else STORE_SCARCITY_PRICES[res]
        )
        for res in STORE_BASE_PRICES
    }

  def get_turn_order(self) -> list[Player]:
    if self.current_round == 1:
      order = list(self.players)
      random.shuffle(order)
      return order
    order = list(self.players)
    random.shuffle(order)
    return sorted(order, key=lambda p: p.base_net_worth())

  def process_food_consumption(self):
    logger.info(
        f"--- STARTING FOOD CONSUMPTION PHASE (ROUND"
        f" {self.current_round}/{self.config.total_rounds}) ---"
    )
    for p in self.players:
      if p.food >= 2:
        p.food -= 2
        p.suffering_food_shortage = False
        msg = f"Consumed 2 Food for workforce. Remaining Food: 🌾{p.food}."
        logger.info(f"[{p.name}] {msg}")
        p.log_event(f"Round {self.current_round} [CONSUMPTION]: {msg}")
      else:
        p.suffering_food_shortage = True
        self.food_shortage_tracker[p.name] = (
            self.food_shortage_tracker.get(p.name, 0) + 1
        )
        msg = (
            f"WORK SHORTAGE! Lacked 2 Food (Has 🌾{p.food}). Sales restricted"
            " this round!"
        )
        logger.warning(f"[{p.name}] {msg}")
        p.log_event(f"Round {self.current_round} [CONSUMPTION]: {msg}")

  def process_land_action(self, player: Player, decision: dict) -> bool:
    action = str(decision.get("action", "PASS")).upper()
    if action == "PASS":
      logger.info(f"[{player.name}] Passed on Land Grant.")
      player.log_event(
          f"Round {self.current_round} [LAND]: Passed on Land Grant."
      )
      return True
    elif action == "CLAIM_TILE":
      tile_id = decision.get("tile_id")
      development = str(decision.get("development", "FOOD")).upper()

      if (
          tile_id is None
          or not isinstance(tile_id, int)
          or not (0 <= tile_id <= 15)
      ):
        logger.warning(
            f"[{player.name}] REJECTED MOVE: Invalid tile_id '{tile_id}'."
        )
        self.telemetry.record_rejected_move(player.name, tile_already_owned=False)
        return False

      tile = self.board[tile_id]
      if tile.owner_id is not None:
        logger.warning(
            f"[{player.name}] REJECTED MOVE: Tile {tile_id} is already owned!"
        )
        self.telemetry.record_rejected_move(player.name, tile_already_owned=True)
        return False

      if development not in ["FOOD", "ENERGY", "CRYSTITE", "NONE"]:
        development = "NONE"

      total_cost = BASE_LAND_CLAIM_FEE + MULE_COSTS.get(development, 0)
      if player.cash < total_cost:
        logger.warning(
            f"[{player.name}] REJECTED MOVE: Cannot afford ${total_cost} for"
            f" Land Claim + {development} M.U.L.E. (Cash: ${player.cash})"
        )
        return False

      player.cash -= total_cost
      tile.owner_id = player.player_id
      player.owned_tiles.append(tile)

      if tile.terrain == "RIVER" and development == "CRYSTITE":
        self.mule_failure_tracker[player.name] = (
            self.mule_failure_tracker.get(player.name, 0) + 1
        )
        msg = (
            f"M.U.L.E. FAILURE! Tried installing Crystite M.U.L.E. in River Tile"
            f" {tile_id}. Lost M.U.L.E.!"
        )
        logger.warning(f"💥 [{player.name}] {msg}")
        player.log_event(f"Round {self.current_round} [LAND]: {msg}")
        tile.development = None
        return True

      if tile.terrain == "PLAIN" and development == "FOOD":
        if random.random() < 0.50:
          self.mule_failure_tracker[player.name] = (
              self.mule_failure_tracker.get(player.name, 0) + 1
          )
          msg = (
              f"M.U.L.E. RUNAWAY! Food M.U.L.E. failed to install on Plain Tile"
              f" {tile_id}. Lost M.U.L.E.!"
          )
          logger.warning(f"💥 [{player.name}] {msg}")
          player.log_event(f"Round {self.current_round} [LAND]: {msg}")
          tile.development = None
          return True

      tile.development = development if development != "NONE" else None
      msg = (
          f"Claimed Tile {tile_id} ({tile.terrain}) with {development} M.U.L.E."
          f" (Paid ${total_cost})."
      )
      logger.info(f"[{player.name}] {msg}")
      player.log_event(f"Round {self.current_round} [LAND]: {msg}")
      return True
    else:
      logger.warning(f"[{player.name}] REJECTED ILLEGAL MOVE: '{action}'")
      return False

  def execute_direct_trade(
      self, proposer: Player, responder: Player, offer: dict, response: dict
  ):
    resp_action = str(response.get("action", "REJECT")).upper()
    tax = proposer.calculate_dan_tax(self)

    trade_type = offer.get("type", "SELL").upper()
    resource = offer.get("resource", "FOOD").upper()

    self.telemetry.record_trade_proposal(proposer.name, responder.name)

    if (
        trade_type == "SELL"
        and proposer.suffering_food_shortage
        and resource in ["ENERGY", "CRYSTITE"]
    ):
      logger.warning(
          f"DIRECT TRADE REJECTED: {proposer.name} is suffering a Workforce Food"
          f" Shortage and cannot sell {resource}!"
      )
      proposer.log_event(
          f"Round {self.current_round} [DIRECT TRADE]: Attempted trade"
          f" blocked by Workforce Shortage."
      )
      return

    self.trade_log.append({
        "round": self.current_round,
        "proposer": proposer.name,
        "responder": responder.name,
        "offer": offer,
        "outcome": resp_action,
    })

    # Tax is charged on REJECT (cost of attempting) but NOT on failed-after-accept trades
    if resp_action != "ACCEPT":
      proposer.cash -= tax  # Rejection tax: cost of making an offer that was turned down
      self.dan_tax_log[(self.current_round, proposer.name, responder.name)] = tax
      self.telemetry.record_trade_outcome(
          proposer.name, responder.name, accepted=False
      )
      logger.info(
          f"[{responder.name}] REJECTED trade offer from {proposer.name} (Paid"
          f" ${tax} Dan Tax)."
      )
      proposer.log_event(
          f"Round {self.current_round} [DIRECT TRADE]: Offer to"
          f" {responder.name} was REJECTED (Paid ${tax} tax)."
      )
      responder.log_event(
          f"Round {self.current_round} [DIRECT TRADE]: REJECTED offer from"
          f" {proposer.name}."
      )
      return

    qty = offer.get("quantity", 0)
    unit_price = offer.get("price_per_unit", 0)
    total_price = qty * unit_price

    if qty <= 0 or unit_price <= 0:
      logger.warning("TRADE CANCELLED: Invalid quantity or price.")
      return

    if trade_type == "SELL":
      proposer_stock = (
          proposer.food
          if resource == "FOOD"
          else (proposer.energy if resource == "ENERGY" else proposer.crystite)
      )

      # V2.1 Fix: Validate BEFORE deducting tax — don't penalize accepted-but-impossible trades
      if proposer_stock < qty:
        fail_msg = f"Proposer {proposer.name} lacks stock (Has {proposer_stock} {resource}, needed {qty})."
        logger.warning(f"❌ DIRECT TRADE FAILED (no tax charged): {fail_msg}")
        proposer.log_event(f"Round {self.current_round} [DIRECT TRADE]: Failed! {fail_msg}")
        responder.log_event(f"Round {self.current_round} [DIRECT TRADE]: Failed! {fail_msg}")
        return

      if responder.cash < total_price:
        fail_msg = f"Responder {responder.name} lacks cash (Has ${responder.cash}, needed ${total_price})."
        logger.warning(f"❌ DIRECT TRADE FAILED (no tax charged): {fail_msg}")
        proposer.log_event(f"Round {self.current_round} [DIRECT TRADE]: Failed! {fail_msg}")
        responder.log_event(f"Round {self.current_round} [DIRECT TRADE]: Failed! {fail_msg}")
        return

      # All validation passed — now deduct tax and execute
      proposer.cash -= tax
      self.dan_tax_log[(self.current_round, proposer.name, responder.name)] = tax

      if resource == "FOOD":
        proposer.food -= qty
        responder.food += qty
      elif resource == "ENERGY":
        proposer.energy -= qty
        responder.energy += qty
      elif resource == "CRYSTITE":
        proposer.crystite -= qty
        responder.crystite += qty

      proposer.cash += total_price
      responder.cash -= total_price

      self.telemetry.record_trade_outcome(
          proposer.name, responder.name, accepted=True
      )
      msg = (
          f"SOLD {qty} {resource} to {responder.name} for ${total_price}"
          f" (${unit_price}/ea) [Paid ${tax} Dan Tax]."
      )
      logger.info(f"🤝 DIRECT TRADE SUCCESS! {proposer.name} {msg}")
      proposer.log_event(f"Round {self.current_round} [DIRECT TRADE]: {msg}")
      responder.log_event(
          f"Round {self.current_round} [DIRECT TRADE]: BOUGHT {qty} {resource}"
          f" from {proposer.name} for ${total_price}."
      )

  def process_store_action(self, player: Player, decision: dict):
    action = str(decision.get("action", "PASS")).upper()
    resource = str(decision.get("resource", "FOOD")).upper()
    qty = decision.get("quantity", 0)

    if action == "PASS" or qty <= 0 or resource not in self.store_stock:
      logger.info(f"[{player.name}] No store trading action taken.")
      player.log_event(
          f"Round {self.current_round} [STORE TRADE]: Took no store action."
      )
      return

    tax = player.calculate_dan_tax(self)
    if player.cash < tax:
      logger.warning(
          f"[{player.name}] STORE TRADE CANCELLED: Cannot afford ${tax} Dan"
          " Tax."
      )
      return

    prices = self.get_store_prices()
    price = prices[resource]

    if action == "BUY_FROM_STORE":
      available_stock = self.store_stock[resource]
      actual_qty = min(qty, available_stock)

      if actual_qty <= 0:
        logger.warning(
            f"[{player.name}] STORE BUY FAILED: Store is OUT OF STOCK for"
            f" {resource}!"
        )
        return

      total_cost = (price * actual_qty) + tax
      if player.cash >= total_cost:
        player.cash -= total_cost
        self.store_stock[resource] -= actual_qty

        if resource == "FOOD":
          player.food += actual_qty
        elif resource == "ENERGY":
          player.energy += actual_qty
        elif resource == "CRYSTITE":
          player.crystite += actual_qty

        msg = (
            f"BOUGHT {actual_qty} {resource} from Store for"
            f" ${price * actual_qty} (${price}/ea) + ${tax} Dan Tax."
        )
        logger.info(f"[{player.name}] {msg}")
        player.log_event(f"Round {self.current_round} [STORE TRADE]: {msg}")
      else:
        logger.warning(
            f"[{player.name}] STORE BUY FAILED: Cannot afford ${total_cost}"
            f" (Cash: ${player.cash})."
        )

    elif action == "SELL_TO_STORE":
      if player.suffering_food_shortage and resource in ["ENERGY", "CRYSTITE"]:
        logger.warning(
            f"[{player.name}] STORE SALE REJECTED: Workforce Food Shortage"
            f" prevents selling {resource}!"
        )
        return

      current_qty = (
          player.food
          if resource == "FOOD"
          else (player.energy if resource == "ENERGY" else player.crystite)
      )

      actual_qty = min(qty, current_qty)
      if actual_qty > 0:
        gross_payout = price * actual_qty

        if gross_payout <= tax:
          logger.warning(
              f"[{player.name}] STORE SALE CANCELLED: Gross payout"
              f" (${gross_payout}) does not cover Dan Tax (${tax})!"
          )
          return

        total_payout = gross_payout - tax
        player.cash += total_payout
        self.store_stock[resource] += actual_qty

        if resource == "FOOD":
          player.food -= actual_qty
        elif resource == "ENERGY":
          player.energy -= actual_qty
        elif resource == "CRYSTITE":
          player.crystite -= actual_qty

        msg = (
            f"SOLD {actual_qty} {resource} to Store for ${gross_payout}"
            f" (${price}/ea) - ${tax} Dan Tax (Net: +${total_payout})."
        )
        logger.info(f"[{player.name}] {msg}")
        player.log_event(f"Round {self.current_round} [STORE TRADE]: {msg}")

  def run_production_phase(self):
    logger.info(
        f"--- STARTING PRODUCTION PHASE (ROUND {self.current_round}/{self.config.total_rounds}) ---"
    )
    for p in self.players:
      # V2.1 Fix: Emergency Solar Ration grants 1 energy PER developed tile (not 1 total)
      # Without this, a player with 3 developed tiles still blacks out 2 of them on a single ration
      developed_tiles = [tile for tile in p.owned_tiles if tile.development is not None]
      if p.energy == 0 and developed_tiles:
        ration_amount = len(developed_tiles)
        p.energy += ration_amount
        logger.info(f"🔋 [{p.name}] EMERGENCY SOLAR RATION: Granted {ration_amount} free Energy ({len(developed_tiles)} developed tiles).")
        p.log_event(f"Round {self.current_round} [PRODUCTION]: Received {ration_amount} Emergency Solar Energy rations.")

      produced_summary = []
      for tile in p.owned_tiles:
        if tile.development is None:
          continue

        if p.energy >= 1:
          p.energy -= 1

          if tile.development == "FOOD":
            yield_amount = (
                4
                if tile.terrain == "RIVER"
                else (2 if tile.terrain == "PLAIN" else 1)
            )
            p.food += yield_amount
            produced_summary.append(f"+{yield_amount} Food (Tile {tile.tile_id})")
            logger.info(
                f"[{p.name}] Tile {tile.tile_id} ({tile.terrain}) produced"
                f" {yield_amount} Food."
            )
          elif tile.development == "ENERGY":
            yield_amount = 2
            p.energy += yield_amount
            produced_summary.append(
                f"+{yield_amount} Energy (Tile {tile.tile_id})"
            )
            logger.info(
                f"[{p.name}] Tile {tile.tile_id} ({tile.terrain}) produced"
                f" {yield_amount} Energy."
            )
          elif tile.development == "CRYSTITE":
            yield_amount = (
                3
                if tile.terrain == "MOUNTAIN"
                else (1 if tile.terrain == "PLAIN" else 0)
            )
            p.crystite += yield_amount
            produced_summary.append(
                f"+{yield_amount} Crystite (Tile {tile.tile_id})"
            )
            logger.info(
                f"[{p.name}] Tile {tile.tile_id} ({tile.terrain}) mined"
                f" {yield_amount} Crystite 💎!"
            )
        else:
          self.blackout_tracker[p.name] = (
              self.blackout_tracker.get(p.name, 0) + 1
          )
          logger.warning(
              f"[{p.name}] BLACKOUT! Not enough Energy to power Tile"
              f" {tile.tile_id}."
          )

      summary_str = (
          ", ".join(produced_summary)
          if produced_summary
          else "No production yields"
      )
      p.log_event(f"Round {self.current_round} [PRODUCTION]: {summary_str}.")


# ==========================================
# 9. MAIN BENCHMARK EXECUTION
# ==========================================
SCORING_EPOCH = "2.4.0"  # Bump whenever RESOURCE_SCORE_VALUES or the penalty changes.


def warn_on_scoring_change():
  """Loudly refuses to let old and new scoring silently share a history file.

  V2.4.0 changed RESOURCE_SCORE_VALUES and the penalty curve. Scores from
  earlier versions are NOT comparable — mixing them in one leaderboard would
  be a straightforward measurement error."""
  doc = load_full_history_document()
  matches = doc.get("matches", [])
  stale = [
      m["match_id"] for m in matches
      if m.get("scoring_epoch", m.get("benchmark_version", "0")) != SCORING_EPOCH
  ]
  if not stale:
    return

  print("\n" + "!" * 75)
  print("  ⚠️  SCORING CHANGE DETECTED")
  print(f"  {len(stale)} match(es) in {HISTORY_FILE} were scored under an older")
  print("  scoring epoch and are NOT comparable to V2.4.0 results.")
  print()
  print("  Changed in V2.4.0:")
  print("    - Resource score values now sit BELOW store prices")
  print(f"      {RESOURCE_SCORE_VALUES}")
  print(f"    - Operational penalty is now '{PENALTY_MODE}' (was saturating linear)")
  print()
  print("  Recommended: archive the old file and start a fresh history so the")
  print("  leaderboard covers one scoring epoch only:")
  print(f"    mv {HISTORY_FILE} match_history_pre_2.4.0.json")
  print("!" * 75 + "\n")
  logger.warning(
      f"[SCORING EPOCH] {len(stale)} pre-{SCORING_EPOCH} matches present "
      "in history; cross-epoch scores are not comparable.",
      extra=INFRA,
  )


def run_preflight_check(engine) -> bool:
  """Pings every agent before the match. Returns True if all are healthy.

  This is the single highest-value addition in V2.3.0: every Gemini outage so
  far (revoked key, wrong timeout units, retired model) would have been caught
  here in about one second instead of after a wasted 12-round match."""
  print("\n🩺 PREFLIGHT HEALTH CHECK")
  print("═" * 75)
  logger.info("--- STARTING PREFLIGHT HEALTH CHECK ---")

  results = {}
  for p in engine.players:
    healthy, detail = p.agent.preflight()
    results[p.name] = (healthy, detail)
    icon = "✅" if healthy else "❌"
    line = f"  {icon} {p.name:<9} {detail}"
    print(line)
    if healthy:
      logger.info(f"[PREFLIGHT OK] {p.name}: {detail}")
    else:
      logger.error(f"[PREFLIGHT FAIL] {p.name}: {detail}", extra=INFRA)

  healthy_names = [n for n, (ok, _) in results.items() if ok]
  dead_names = [n for n, (ok, _) in results.items() if not ok]

  print("═" * 75)
  if dead_names:
    print(f"  ⚠️  {len(dead_names)} of {len(results)} agents FAILED preflight: "
          f"{', '.join(dead_names)}")
    print("     Their scores this match will not be meaningful.")
    print("     See the errors log for the categorized cause and suggested fix:")
    print(f"     {ERROR_LOG_PATH}")
    logger.error(
        f"[PREFLIGHT SUMMARY] {len(healthy_names)}/{len(results)} healthy. "
        f"Dead: {', '.join(dead_names)}",
        extra=INFRA,
    )
  else:
    print(f"  All {len(results)} agents responded normally. Starting match.")
    logger.info(f"[PREFLIGHT SUMMARY] All {len(results)} agents healthy.")
  print("═" * 75 + "\n")

  # Reset telemetry so preflight pings don't pollute the match statistics.
  engine.telemetry = TelemetryTracker([p.name for p in engine.players])
  for p in engine.players:
    p.agent.last_error_category = None

  return len(dead_names) == 0


def log_diagnostic_summary(engine):
  """End-of-match reliability table. Answers 'was this a fair match?' at a glance."""
  logger.info("--- MATCH DIAGNOSTIC SUMMARY ---")
  print("\n🔎 RELIABILITY REPORT")
  print("═" * 75)
  print(f"  {'PLAYER':<9} {'HEALTH':<18} {'CALLS':>6} {'ERR%':>6} {'AVG ms':>8}  MODEL")
  print("  " + "─" * 71)

  compromised = []
  for p in engine.players:
    s = engine.telemetry.summary_for_player(p.name)
    model = getattr(p.agent, "model_name", "?")
    print(
        f"  {p.name:<9} {s['health_status']:<18} {s['total_api_calls']:>6} "
        f"{s['error_rate_pct']:>5}% {s['avg_latency_ms']:>8}  {model}"
    )
    logger.info(
        f"[DIAGNOSTIC] {p.name}: health={s['health_status']} "
        f"calls={s['total_api_calls']} error_rate={s['error_rate_pct']}% "
        f"errors={s['error_breakdown']} "
        f"voluntary_pass={s['voluntary_pass_count']} "
        f"fallback_pass={s['fallback_pass_count']} model={model}"
    )
    if s["health_status"] in ("DEAD", "SEVERELY_DEGRADED"):
      compromised.append((p.name, s))

  print("  " + "─" * 71)
  if compromised:
    print(f"  ⚠️  MATCH INTEGRITY WARNING")
    for name, s in compromised:
      causes = ", ".join(f"{k}×{v}" for k, v in s["error_breakdown"].items())
      print(f"     {name}: {s['error_rate_pct']}% of calls failed ({causes})")
    print("     Treat this match's rankings as provisional.")
    logger.warning(
        "[MATCH INTEGRITY] Compromised agents: "
        f"{', '.join(n for n, _ in compromised)}",
        extra=INFRA,
    )
  else:
    print("  ✅ All agents healthy — results are a valid benchmark.")
  print("═" * 75 + "\n")


def run_game():
  mode = "VARIABLE RULES" if ENABLE_VARIABLE_RULES else "FIXED RULES"
  print(
      f"\n⚔️ STARTING IRATA-BENCH V{BENCHMARK_VERSION} "
      f"({mode} · {ROSTER_MODE} ROSTER)\n"
  )
  verify_version_sync()
  warn_on_scoring_change()
  engine = Engine()

  # Announce the rule set drawn for THIS match. Players see the same information
  # inside their prompts; this is the human-facing copy.
  cfg = engine.config
  print("🎲 THIS MATCH'S RULE SET")
  print("═" * 75)
  print(f"  Roster:          {ROSTER_MODE}  ({', '.join(p.name for p in engine.players)})")
  print(f"  Board seed:      {engine.board_seed}")
  print(f"  Rounds:          {cfg.total_rounds}")
  print(f"  Trade rounds:    {cfg.trade_rounds}  (every {cfg.trade_interval})")
  print(f"  Starting cash:   ${cfg.starting_cash} each")
  if cfg.dan_tax_percentage <= 0:
    print("  Dan Tax:         NONE (free trades)")
  else:
    print(f"  Dan Tax:         {cfg.dan_tax_percentage:.1%} of {cfg.tax_basis_description()}")
  print(f"  Scarcity levels: {cfg.scarcity_thresholds}")
  print("═" * 75)
  logger.info(f"[MATCH CONFIG] {cfg.to_dict()} | roster={ROSTER_MODE} | board_seed={engine.board_seed}")

  if RUN_PREFLIGHT_CHECK:
    all_healthy = run_preflight_check(engine)
    if not all_healthy and ABORT_ON_PREFLIGHT_FAIL:
      logger.error("Aborting: ABORT_ON_PREFLIGHT_FAIL is True and agents failed.")
      print("🛑 Aborting before the match. Set ABORT_ON_PREFLIGHT_FAIL=False "
            "to play anyway with degraded agents.")
      finalize_logs()
      return

  # Holt-Laury risk calibration (once per model, cached across matches).
  engine.holt_laury = run_holt_laury_calibration(engine)

  # Load any inter-match memos each model left for itself last time.
  if ENABLE_INTER_MATCH_MEMO and os.path.exists(INTER_MATCH_MEMO_FILE):
    try:
      memo_cache = json.load(open(INTER_MATCH_MEMO_FILE))
      for p in engine.players:
        mid = getattr(p.agent, "model_name", p.name)
        if mid in memo_cache:
          engine.incoming_memos[p.name] = memo_cache[mid]
    except Exception as e:
      logger.debug(f"Could not load inter-match memos: {e}")

  # 1. Pre-Game Interviews
  print("\n🎤 PRE-GAME INTERVIEWS (<50 Words):")
  print("═" * 75)
  for p in engine.players:
    run_pregame_interview(p, engine)

  # 2. Variable-length Tournament Loop (rounds set by this match's config)
  cfg = engine.config
  for r in range(1, cfg.total_rounds + 1):
    engine.current_round = r
    turn_order = engine.get_turn_order()

    logger.info(
        f"Round {r}/{cfg.total_rounds} Priority Order (Lowest Base Wealth / Random"
        f" Tie): {[p.name for p in turn_order]}"
    )

    # Food Consumption Phase
    engine.process_food_consumption()

    # Land Grant Phase
    if not engine.all_tiles_claimed():
      for p in turn_order:
        decision = get_land_decision(p, engine)
        engine.process_land_action(p, decision)
        if LOG_STRUCTURED_DECISIONS:
          engine.record_decision(p, r, "LAND", decision)
      render_ascii_dashboard(engine, "LAND GRANT COMPLETE")
    else:
      logger.info("ALL TILES OWNED: Skipping Land Grant Phase.")

    # Production Phase (With Emergency Solar Rations!)
    engine.run_production_phase()
    render_ascii_dashboard(engine, "PRODUCTION COMPLETE")

    # Mid-game net worth snapshot: captured AFTER production (value created) but
    # BEFORE any selling this round, so the timeline reflects productive wealth.
    if SNAPSHOT_NET_WORTH_PER_ROUND:
      for p in engine.players:
        engine.net_worth_timeline[p.name].append(p.net_worth(engine))

    # Direct Trade Phase (on this match's trade rounds)
    if r % cfg.trade_interval == 0:
      logger.info(f"--- STARTING DIRECT TRADES (ROUND {r}) ---")
      for p_proposer in turn_order:
        tax = p_proposer.calculate_dan_tax(engine)
        if p_proposer.cash < tax:
          logger.info(
              f"[{p_proposer.name}] Auto-passing Direct Trade (Cannot afford"
              f" ${tax} Dan Tax)."
          )
          p_proposer.log_event(
              f"Round {r} [DIRECT TRADE]: Auto-passed (Cannot afford ${tax}"
              " tax)."
          )
          continue

        opponents = [p for p in engine.players if p != p_proposer]
        if opponents:
          p_responder = random.choice(opponents)
          offer = get_direct_trade_offer(p_proposer, p_responder, engine)
          if str(offer.get("action", "PASS")).upper() == "PROPOSE_TRADE":
            logger.info(f"[{p_proposer.name}] PROPOSED TRADE: {offer}")
            if LOG_STRUCTURED_DECISIONS:
              engine.record_decision(p_proposer, r, "TRADE", offer)
            response = get_direct_trade_response(
                p_responder, p_proposer, offer, engine
            )
            logger.info(f"[{p_responder.name}] RESPONSE: {response}")
            if LOG_STRUCTURED_DECISIONS:
              engine.record_decision(p_responder, r, "TRADE", response)
            engine.execute_direct_trade(
                p_proposer, p_responder, offer, response
            )
          else:
            p_proposer.log_event(
                f"Round {r} [DIRECT TRADE]: Passed on proposal."
            )
      render_ascii_dashboard(engine, "DIRECT TRADES COMPLETE")
    else:
      logger.info(f"SKIPPING DIRECT TRADES (Round {r}).")

    # Store Trade Phase (on this match's trade rounds)
    if r % cfg.trade_interval == 0:
      logger.info(f"--- STARTING STORE TRADES (ROUND {r}) ---")
      for p in turn_order:
        tax = p.calculate_dan_tax(engine)
        # Only auto-pass if tax is non-zero AND player genuinely can't afford it.
        # If tax == 0 (poverty waiver active), always let them through to trade tax-free.
        if tax > 0 and p.cash < tax:
          logger.info(
              f"[{p.name}] Auto-passing Store Trade (Cannot afford ${tax} Dan Tax)."
          )
          p.log_event(
              f"Round {r} [STORE TRADE]: Auto-passed (Cannot afford ${tax} tax)."
          )
          continue
        decision = get_store_decision(p, engine)
        engine.process_store_action(p, decision)
        if LOG_STRUCTURED_DECISIONS:
          engine.record_decision(p, r, "STORE", decision)
        if TRACK_EV_POSITIVE_PASSES:
          engine.analyze_ev_pass(p, decision)
      render_ascii_dashboard(engine, "STORE TRADE COMPLETE")
    else:
      logger.info(f"SKIPPING STORE TRADES (Round {r}).")

    # Real-time Autosave Checkpoint
    save_turn_checkpoint(engine)

  # 3. Match Summary & Post-Game Interviews
  winner = max(engine.players, key=lambda p: p.net_worth(engine))
  print("\n" + "🏆" * 35)
  print(
      f"  {cfg.total_rounds}-ROUND TOURNAMENT COMPLETE! WINNER: {winner.name}"
      f" (${winner.net_worth(engine)})"
  )
  print("🏆" * 35 + "\n")

  # Permanent Match Save
  # Post-game interviews FIRST (so they can be persisted into the record).
  print("\n🎤 POST-GAME INTERVIEWS (<50 Words):")
  print("═" * 75)
  for p in engine.players:
    raw_quote = get_spicy_post_game_statement(p, engine)
    if LOG_VERBATIM_INTERVIEWS:
      engine.postgame_interviews[p.name] = raw_quote
    badges = {1: "🔴", 2: "🔵", 3: "🟢", 4: "🟡"}
    badge = badges.get(p.player_id, "⚪")
    wins, losses = get_player_win_loss_stats(p.name)

    wrapped_quote = textwrap.fill(
        f'"{raw_quote}"',
        width=70,
        initial_indent="   ",
        subsequent_indent="   ",
    )
    print(f"{badge} [{p.name} ({wins} Wins / {losses} Losses)]:\n{wrapped_quote}\n")

  # Now persist the record (includes interviews, timeline, decision ledger).
  save_match_record(engine)

  # Reliability report: makes it obvious when a "result" was really an outage.
  log_diagnostic_summary(engine)

  # Net-worth-over-time chart for this match.
  if RENDER_TIMELINE_CHART and SNAPSHOT_NET_WORTH_PER_ROUND:
    try:
      chart_path = render_net_worth_timeline(engine)
      if chart_path:
        print(f"📈 Net worth timeline: {chart_path}")
    except Exception as e:
      logger.debug(f"Timeline chart failed: {e}")

  # Operator-only reveal of the private notes each player wrote to itself.
  # (Opponents never saw these during the match.)
  if ENABLE_SECRET_NOTES and any(p.secret_notes for p in engine.players):
    print("\n🤫 SECRET NOTES (revealed to operator only — hidden from opponents during play)")
    print("═" * 75)
    badges = {1: "🔴", 2: "🔵", 3: "🟢", 4: "🟡"}
    for p in engine.players:
      if not p.secret_notes:
        continue
      print(f"{badges.get(p.player_id,'⚪')} {p.name}:")
      for n in p.secret_notes:
        wrapped = textwrap.fill(
            n["note"], width=66,
            initial_indent=f"   R{n['round']} [{n['phase']}]: ",
            subsequent_indent="      ")
        print(wrapped)
      print()

  # Persist inter-match memos (keyed by model id, so a model gets its own memo
  # next match regardless of seat). Written after post-game interviews collected.
  if ENABLE_INTER_MATCH_MEMO:
    outgoing = getattr(engine, "outgoing_memos", {})
    if outgoing:
      try:
        cache = json.load(open(INTER_MATCH_MEMO_FILE)) if os.path.exists(INTER_MATCH_MEMO_FILE) else {}
      except Exception:
        cache = {}
      print("\n📝 MEMOS TO NEXT SELF (private directives, revealed to operator)")
      print("═" * 75)
      badges = {1: "🔴", 2: "🔵", 3: "🟢", 4: "🟡"}
      for p in engine.players:
        if p.name in outgoing:
          mid = getattr(p.agent, "model_name", p.name)
          cache[mid] = outgoing[p.name]
          wrapped = textwrap.fill(
              outgoing[p.name], width=68,
              initial_indent=f"{badges.get(p.player_id,'⚪')} {p.name}: ",
              subsequent_indent="   ")
          print(wrapped)
      try:
        with open(INTER_MATCH_MEMO_FILE, "w") as f:
          json.dump(cache, f, indent=2)
      except Exception as e:
        logger.warning(f"Could not save inter-match memos: {e}")

  print(f"📂 Full log:   {MAIN_LOG_PATH}")
  print(f"📂 Errors only: {ERROR_LOG_PATH}")
  finalize_logs()


if __name__ == "__main__":
  try:
    run_game()
  except EnvironmentError as e:
    # Missing API keys are a config problem, not a crash — say so plainly.
    logger.error(f"Startup configuration error: {e}")
    print(f"\n🛑 CONFIGURATION ERROR\n{e}\n")
    finalize_logs()
    sys.exit(1)
  except KeyboardInterrupt:
    logger.warning("Run interrupted by user (Ctrl-C).")
    print("\n⏹️  Interrupted. Partial log written.")
    finalize_logs()
    sys.exit(130)
  except Exception as e:
    logger.error(f"Unhandled exception: {type(e).__name__}: {e}", exc_info=True)
    print(f"\n💥 UNEXPECTED ERROR: {type(e).__name__}: {e}")
    print(f"   Full traceback written to {ERROR_LOG_PATH}")
    finalize_logs()
    sys.exit(1)