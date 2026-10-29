"""SQLite persistence layer.

One database is the single source of truth for a run: discovered candidates,
website verdicts, scored leads, LLM telemetry, task checkpoints and the HTTP
response cache. Export never recomputes anything — it just reads this.

Concurrency: one connection per thread (WAL mode), so the agent thread pool can
write safely without a global lock.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .models import (
    Candidate,
    Cell,
    CriticReview,
    Lead,
    LeadStatus,
    Qualification,
    RunStats,
    WebsiteVerdict,
    utcnow,
)

SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS runs (
  run_id       TEXT PRIMARY KEY,
  created_at   TEXT NOT NULL,
  finished_at  TEXT,
  status       TEXT NOT NULL,
  niche        TEXT,
  state        TEXT,
  config_json  TEXT,
  stats_json   TEXT,
  report_path  TEXT
);

CREATE TABLE IF NOT EXISTS tasks (
  run_id     TEXT NOT NULL,
  stage      TEXT NOT NULL,
  cell_key   TEXT NOT NULL,
  status     TEXT NOT NULL DEFAULT 'pending',
  attempts   INTEGER NOT NULL DEFAULT 0,
  error      TEXT,
  updated_at TEXT NOT NULL,
  PRIMARY KEY (run_id, stage, cell_key)
);

CREATE TABLE IF NOT EXISTS candidates (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id      TEXT NOT NULL,
  dedupe_key  TEXT NOT NULL,
  name        TEXT NOT NULL,
  city        TEXT,
  state       TEXT,
  zip         TEXT,
  phone       TEXT,
  address     TEXT,
  category    TEXT,
  rating      REAL,
  reviews     INTEGER,
  website     TEXT,
  source      TEXT,
  source_url  TEXT,
  evidence    TEXT,
  created_at  TEXT NOT NULL,
  UNIQUE (run_id, dedupe_key)
);

CREATE TABLE IF NOT EXISTS verdicts (
  candidate_id INTEGER PRIMARY KEY REFERENCES candidates(id) ON DELETE CASCADE,
  has_website  INTEGER,
  website_url  TEXT,
  confidence   REAL,
  method       TEXT,
  evidence     TEXT,
  ambiguous    INTEGER NOT NULL DEFAULT 0,
  verified_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS leads (
  candidate_id    INTEGER PRIMARY KEY REFERENCES candidates(id) ON DELETE CASCADE,
  run_id          TEXT NOT NULL,
  fit             INTEGER,
  tier            TEXT,
  qualifier_reason TEXT,
  signals_json    TEXT,
  critic_status   TEXT,
  critic_notes    TEXT,
  status          TEXT NOT NULL,
  updated_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS llm_calls (
  id                INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id            TEXT,
  agent             TEXT,
  model             TEXT,
  prompt_tokens     INTEGER,
  completion_tokens INTEGER,
  latency_ms        INTEGER,
  ok                INTEGER,
  error             TEXT,
  created_at        TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
  id       INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id   TEXT,
  ts       TEXT NOT NULL,
  level    TEXT NOT NULL,
  agent    TEXT,
  message  TEXT NOT NULL,
  data_json TEXT
);

CREATE TABLE IF NOT EXISTS http_cache (
  key        TEXT PRIMARY KEY,
  provider   TEXT,
  url        TEXT,
  payload    TEXT,
  created_at TEXT NOT NULL,
  expires_at TEXT
);

CREATE TABLE IF NOT EXISTS suppression (
  dedupe_key TEXT PRIMARY KEY,
  phone      TEXT,
  reason     TEXT,
  created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_candidates_run ON candidates(run_id);
CREATE INDEX IF NOT EXISTS idx_leads_run ON leads(run_id);
CREATE INDEX IF NOT EXISTS idx_leads_tier ON leads(run_id, tier);
CREATE INDEX IF NOT EXISTS idx_tasks_run ON tasks(run_id);
CREATE INDEX IF NOT EXISTS idx_events_run ON events(run_id);
"""


class Database:
    """Thread-local SQLite connection manager with WAL and sane pragmas."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self.init_schema()

    def conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, timeout=30.0, isolation_level=None)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA busy_timeout=30000")
            self._local.conn = conn
        return conn

    def init_schema(self) -> None:
        self.conn().executescript(SCHEMA)

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        conn = self.conn()
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

    def query(self, sql: str, params: Iterable[Any] = ()) -> list[sqlite3.Row]:
        return list(self.conn().execute(sql, tuple(params)).fetchall())

    def one(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Row | None:
        return self.conn().execute(sql, tuple(params)).fetchone()

    def execute(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Cursor:
        return self.conn().execute(sql, tuple(params))

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None


class Store:
    """High-level repository. All writes are idempotent."""

    def __init__(self, db: Database) -> None:
        self.db = db

    # ------------------------------------------------------------------ runs
    def create_run(self, run_id: str, niche: str, state: str, config: dict[str, Any]) -> None:
        self.db.execute(
            "INSERT OR IGNORE INTO runs (run_id, created_at, status, niche, state, config_json) "
            "VALUES (?,?,?,?,?,?)",
            (run_id, utcnow(), "running", niche, state, json.dumps(config, default=str)),
        )

    def finish_run(self, run_id: str, status: str, stats: RunStats, report_path: str | None) -> None:
        self.db.execute(
            "UPDATE runs SET finished_at=?, status=?, stats_json=?, report_path=? WHERE run_id=?",
            (utcnow(), status, stats.model_dump_json(), report_path, run_id),
        )

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        row = self.db.one("SELECT * FROM runs WHERE run_id=?", (run_id,))
        return dict(row) if row else None

    def latest_run(self) -> dict[str, Any] | None:
        row = self.db.one("SELECT * FROM runs ORDER BY created_at DESC LIMIT 1")
        return dict(row) if row else None

    def list_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = self.db.query("SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,))
        return [dict(r) for r in rows]

    # ----------------------------------------------------------------- tasks
    def seed_task(self, run_id: str, stage: str, cell_key: str) -> None:
        self.db.execute(
            "INSERT OR IGNORE INTO tasks (run_id, stage, cell_key, status, updated_at) "
            "VALUES (?,?,?,'pending',?)",
            (run_id, stage, cell_key, utcnow()),
        )

    def task_state(self, run_id: str, stage: str, cell_key: str) -> str:
        row = self.db.one(
            "SELECT status FROM tasks WHERE run_id=? AND stage=? AND cell_key=?",
            (run_id, stage, cell_key),
        )
        return row["status"] if row else "pending"

    def start_task(self, run_id: str, stage: str, cell_key: str) -> None:
        self.db.execute(
            "INSERT INTO tasks (run_id, stage, cell_key, status, attempts, updated_at) "
            "VALUES (?,?,?,'running',1,?) "
            "ON CONFLICT(run_id, stage, cell_key) DO UPDATE SET "
            "status='running', attempts=attempts+1, error=NULL, updated_at=excluded.updated_at",
            (run_id, stage, cell_key, utcnow()),
        )

    def complete_task(self, run_id: str, stage: str, cell_key: str) -> None:
        self.db.execute(
            "UPDATE tasks SET status='done', updated_at=? "
            "WHERE run_id=? AND stage=? AND cell_key=?",
            (utcnow(), run_id, stage, cell_key),
        )

    def fail_task(self, run_id: str, stage: str, cell_key: str, error: str) -> None:
        self.db.execute(
            "UPDATE tasks SET status='failed', error=?, updated_at=? "
            "WHERE run_id=? AND stage=? AND cell_key=?",
            (error[:500], utcnow(), run_id, stage, cell_key),
        )

    def task_counts(self, run_id: str) -> dict[str, dict[str, int]]:
        rows = self.db.query(
            "SELECT stage, status, COUNT(*) n FROM tasks WHERE run_id=? GROUP BY stage, status",
            (run_id,),
        )
        out: dict[str, dict[str, int]] = {}
        for r in rows:
            out.setdefault(r["stage"], {})[r["status"]] = r["n"]
        return out

    # ------------------------------------------------------------ candidates
    def upsert_candidate(self, run_id: str, cand: Candidate) -> int:
        """Insert-or-ignore by dedupe key; returns the candidate id."""
        from .tools.extract import dedupe_key  # local import avoids a cycle

        key = dedupe_key(cand.name, cand.city, cand.zip)
        self.db.execute(
            "INSERT INTO candidates (run_id, dedupe_key, name, city, state, zip, phone, address,"
            " category, rating, reviews, website, source, source_url, evidence, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(run_id, dedupe_key) DO UPDATE SET "
            "rating = COALESCE(candidates.rating, excluded.rating), "
            "reviews = COALESCE(candidates.reviews, excluded.reviews), "
            "phone = COALESCE(candidates.phone, excluded.phone), "
            "address = COALESCE(candidates.address, excluded.address), "
            "website = COALESCE(candidates.website, excluded.website), "
            "evidence = CASE WHEN length(excluded.evidence) > length(candidates.evidence) "
            "           THEN excluded.evidence ELSE candidates.evidence END",
            (
                run_id, key, cand.name, cand.city, cand.state, cand.zip, cand.phone,
                cand.address, cand.category, cand.rating, cand.reviews, cand.website,
                cand.source, cand.source_url, cand.evidence, utcnow(),
            ),
        )
        row = self.db.one(
            "SELECT id FROM candidates WHERE run_id=? AND dedupe_key=?", (run_id, key)
        )
        assert row is not None
        return int(row["id"])

    def candidates(self, run_id: str) -> list[dict[str, Any]]:
        rows = self.db.query("SELECT * FROM candidates WHERE run_id=? ORDER BY id", (run_id,))
        return [dict(r) for r in rows]

    def candidates_pending_verdict(self, run_id: str) -> list[dict[str, Any]]:
        rows = self.db.query(
            "SELECT c.* FROM candidates c LEFT JOIN verdicts v ON v.candidate_id = c.id "
            "WHERE c.run_id=? AND v.candidate_id IS NULL ORDER BY c.id",
            (run_id,),
        )
        return [dict(r) for r in rows]

    def is_suppressed(self, key: str) -> bool:
        return self.db.one("SELECT 1 FROM suppression WHERE dedupe_key=?", (key,)) is not None

    def suppress(self, key: str, phone: str | None, reason: str) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO suppression (dedupe_key, phone, reason, created_at) "
            "VALUES (?,?,?,?)",
            (key, phone, reason, utcnow()),
        )

    # -------------------------------------------------------------- verdicts
    def set_verdict(self, candidate_id: int, verdict: WebsiteVerdict) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO verdicts (candidate_id, has_website, website_url, confidence,"
            " method, evidence, ambiguous, verified_at) VALUES (?,?,?,?,?,?,?,?)",
            (
                candidate_id,
                int(verdict.has_website),
                verdict.website_url,
                verdict.confidence,
                verdict.method.value,
                verdict.evidence[:2000],
                int(verdict.ambiguous),
                utcnow(),
            ),
        )

    def get_verdict(self, candidate_id: int) -> dict[str, Any] | None:
        row = self.db.one("SELECT * FROM verdicts WHERE candidate_id=?", (candidate_id,))
        return dict(row) if row else None

    # ----------------------------------------------------------------- leads
    def upsert_lead(
        self,
        run_id: str,
        candidate_id: int,
        status: LeadStatus,
        qualification: Qualification | None = None,
        critic: CriticReview | None = None,
    ) -> None:
        existing = self.db.one("SELECT * FROM leads WHERE candidate_id=?", (candidate_id,))
        fit = qualification.fit if qualification else (existing["fit"] if existing else None)
        tier = qualification.tier.value if qualification else (existing["tier"] if existing else None)
        reason = (
            qualification.reason
            if qualification
            else (existing["qualifier_reason"] if existing else "")
        )
        signals = (
            json.dumps(qualification.signals, default=str)
            if qualification
            else (existing["signals_json"] if existing else None)
        )
        if critic is not None:
            critic_status = "accepted" if critic.accepted else "rejected"
            critic_notes = "; ".join(critic.issues) or critic.rationale
        else:
            critic_status = existing["critic_status"] if existing else None
            critic_notes = existing["critic_notes"] if existing else ""
        self.db.execute(
            "INSERT INTO leads (candidate_id, run_id, fit, tier, qualifier_reason, signals_json,"
            " critic_status, critic_notes, status, updated_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(candidate_id) DO UPDATE SET fit=excluded.fit, tier=excluded.tier, "
            "qualifier_reason=excluded.qualifier_reason, signals_json=excluded.signals_json, "
            "critic_status=COALESCE(excluded.critic_status, leads.critic_status), "
            "critic_notes=COALESCE(excluded.critic_notes, leads.critic_notes), "
            "status=excluded.status, updated_at=excluded.updated_at",
            (
                candidate_id, run_id, fit, tier, reason, signals, critic_status,
                critic_notes, status.value, utcnow(),
            ),
        )

    def leads(self, run_id: str) -> list[dict[str, Any]]:
        """Joined lead view used by export and reporting."""
        rows = self.db.query(
            """
            SELECT
              c.id AS candidate_id, l.run_id, c.name, c.city, c.state, c.zip, c.phone,
              c.address, c.category, c.rating, c.reviews, c.source, c.source_url,
              v.has_website, v.website_url, v.confidence AS website_confidence,
              v.method AS website_method,
              l.fit, l.tier, l.qualifier_reason, l.critic_status, l.critic_notes, l.status
            FROM leads l
            JOIN candidates c ON c.id = l.candidate_id
            LEFT JOIN verdicts v ON v.candidate_id = c.id
            WHERE l.run_id = ?
            ORDER BY
              CASE l.tier WHEN 'A' THEN 0 WHEN 'B' THEN 1 WHEN 'C' THEN 2 ELSE 3 END,
              l.fit DESC, c.name
            """,
            (run_id,),
        )
        return [dict(r) for r in rows]

    def counts(self, run_id: str) -> dict[str, Any]:
        def scalar(sql: str, params: Iterable[Any] = ()) -> Any:
            row = self.db.one(sql, params)
            return row[0] if row else 0

        return {
            "candidates": scalar("SELECT COUNT(*) FROM candidates WHERE run_id=?", (run_id,)),
            "verdicts": scalar(
                "SELECT COUNT(*) FROM verdicts v JOIN candidates c ON c.id=v.candidate_id "
                "WHERE c.run_id=?",
                (run_id,),
            ),
            "with_website": scalar(
                "SELECT COUNT(*) FROM verdicts v JOIN candidates c ON c.id=v.candidate_id "
                "WHERE c.run_id=? AND v.has_website=1",
                (run_id,),
            ),
            "without_website": scalar(
                "SELECT COUNT(*) FROM verdicts v JOIN candidates c ON c.id=v.candidate_id "
                "WHERE c.run_id=? AND v.has_website=0",
                (run_id,),
            ),
            "leads": scalar("SELECT COUNT(*) FROM leads WHERE run_id=?", (run_id,)),
            "tier_a": scalar("SELECT COUNT(*) FROM leads WHERE run_id=? AND tier='A'", (run_id,)),
            "tier_b": scalar("SELECT COUNT(*) FROM leads WHERE run_id=? AND tier='B'", (run_id,)),
            "llm_calls": scalar("SELECT COUNT(*) FROM llm_calls WHERE run_id=?", (run_id,)),
            "llm_tokens": scalar(
                "SELECT COALESCE(SUM(COALESCE(prompt_tokens,0)+COALESCE(completion_tokens,0)),0) "
                "FROM llm_calls WHERE run_id=?",
                (run_id,),
            ),
        }

    # ----------------------------------------------------------- telemetry
    def record_llm_call(
        self,
        run_id: str | None,
        agent: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: int,
        ok: bool,
        error: str | None = None,
    ) -> None:
        self.db.execute(
            "INSERT INTO llm_calls (run_id, agent, model, prompt_tokens, completion_tokens,"
            " latency_ms, ok, error, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (
                run_id, agent, model, prompt_tokens, completion_tokens, latency_ms,
                int(ok), (error or "")[:500] or None, utcnow(),
            ),
        )

    def log_event(
        self, run_id: str | None, level: str, agent: str, message: str, data: dict[str, Any] | None = None
    ) -> None:
        self.db.execute(
            "INSERT INTO events (run_id, ts, level, agent, message, data_json) VALUES (?,?,?,?,?,?)",
            (run_id, utcnow(), level, agent, message[:1000], json.dumps(data, default=str) if data else None),
        )

    # ----------------------------------------------------------------- cache
    def cache_get(self, key: str) -> dict[str, Any] | None:
        row = self.db.one("SELECT payload, expires_at FROM http_cache WHERE key=?", (key,))
        if not row:
            return None
        if row["expires_at"] and row["expires_at"] < utcnow():
            return None
        try:
            return json.loads(row["payload"])
        except json.JSONDecodeError:
            return None

    def cache_put(self, key: str, provider: str, url: str, payload: Any, ttl_hours: int) -> None:
        from datetime import datetime, timedelta, timezone

        expires = (
            (datetime.now(timezone.utc) + timedelta(hours=ttl_hours)).isoformat(timespec="seconds")
            if ttl_hours > 0
            else None
        )
        self.db.execute(
            "INSERT OR REPLACE INTO http_cache (key, provider, url, payload, created_at, expires_at)"
            " VALUES (?,?,?,?,?,?)",
            (key, provider, url, json.dumps(payload, default=str), utcnow(), expires),
        )
