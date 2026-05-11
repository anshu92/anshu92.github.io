from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable

from . import memory
from .models import SourceItem


SCHEMA_VERSION = 1


def db_path(path: str | Path | None = None) -> Path:
    return Path(path) if path else memory.DATA / "items.sqlite"


def connect(path: str | Path | None = None) -> sqlite3.Connection:
    memory.ensure_dirs()
    conn = sqlite3.connect(str(db_path(path)))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    migrate(conn)
    return conn


def migrate(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS items (
            rowid INTEGER PRIMARY KEY,
            item_id TEXT UNIQUE NOT NULL,
            canonical_url TEXT NOT NULL,
            source_kind TEXT NOT NULL,
            source_name TEXT NOT NULL,
            source_tier INTEGER NOT NULL DEFAULT 2,
            title TEXT NOT NULL,
            authors_json TEXT NOT NULL DEFAULT '[]',
            published_at TEXT,
            updated_at TEXT,
            doi TEXT NOT NULL DEFAULT '',
            arxiv_id TEXT NOT NULL DEFAULT '',
            venue_or_blog TEXT NOT NULL DEFAULT '',
            abstract_or_excerpt TEXT NOT NULL DEFAULT '',
            body_text TEXT NOT NULL DEFAULT '',
            tags_json TEXT NOT NULL DEFAULT '[]',
            extra_json TEXT NOT NULL DEFAULT '{}',
            topic_scores_json TEXT NOT NULL DEFAULT '{}',
            quality_signals_json TEXT NOT NULL DEFAULT '{}',
            citation_signals_json TEXT NOT NULL DEFAULT '{}',
            daily_score REAL NOT NULL DEFAULT 0,
            deep_dive_score REAL NOT NULL DEFAULT 0,
            status_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_row_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE UNIQUE INDEX IF NOT EXISTS idx_items_doi ON items(doi) WHERE doi != '';
        CREATE UNIQUE INDEX IF NOT EXISTS idx_items_arxiv ON items(arxiv_id) WHERE arxiv_id != '';
        CREATE VIRTUAL TABLE IF NOT EXISTS items_fts
        USING fts5(title, abstract_or_excerpt, body_text, content='items', content_rowid='rowid');
        CREATE TRIGGER IF NOT EXISTS items_ai AFTER INSERT ON items BEGIN
            INSERT INTO items_fts(rowid, title, abstract_or_excerpt, body_text)
            VALUES (new.rowid, new.title, new.abstract_or_excerpt, new.body_text);
        END;
        CREATE TRIGGER IF NOT EXISTS items_ad AFTER DELETE ON items BEGIN
            INSERT INTO items_fts(items_fts, rowid, title, abstract_or_excerpt, body_text)
            VALUES('delete', old.rowid, old.title, old.abstract_or_excerpt, old.body_text);
        END;
        CREATE TRIGGER IF NOT EXISTS items_au AFTER UPDATE ON items BEGIN
            INSERT INTO items_fts(items_fts, rowid, title, abstract_or_excerpt, body_text)
            VALUES('delete', old.rowid, old.title, old.abstract_or_excerpt, old.body_text);
            INSERT INTO items_fts(rowid, title, abstract_or_excerpt, body_text)
            VALUES (new.rowid, new.title, new.abstract_or_excerpt, new.body_text);
        END;
        """
    )
    conn.execute(
        "INSERT OR REPLACE INTO meta(key, value) VALUES('schema_version', ?)",
        (str(SCHEMA_VERSION),),
    )
    conn.commit()


def upsert_items(conn: sqlite3.Connection, items: Iterable[SourceItem]) -> int:
    count = 0
    for raw in items:
        item = raw.normalized()
        conn.execute(
            """
            INSERT INTO items (
                item_id, canonical_url, source_kind, source_name, source_tier, title,
                authors_json, published_at, updated_at, doi, arxiv_id, venue_or_blog,
                abstract_or_excerpt, body_text, tags_json, extra_json, updated_row_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(item_id) DO UPDATE SET
                canonical_url=excluded.canonical_url,
                source_kind=excluded.source_kind,
                source_name=excluded.source_name,
                source_tier=excluded.source_tier,
                title=excluded.title,
                authors_json=excluded.authors_json,
                published_at=COALESCE(excluded.published_at, items.published_at),
                updated_at=COALESCE(excluded.updated_at, items.updated_at),
                doi=excluded.doi,
                arxiv_id=excluded.arxiv_id,
                venue_or_blog=excluded.venue_or_blog,
                abstract_or_excerpt=excluded.abstract_or_excerpt,
                body_text=excluded.body_text,
                tags_json=excluded.tags_json,
                extra_json=excluded.extra_json,
                updated_row_at=CURRENT_TIMESTAMP
            """,
            (
                item.item_id,
                item.canonical_url,
                item.source_kind,
                item.source_name,
                item.source_tier,
                item.title,
                json.dumps([a.model_dump() for a in item.authors]),
                item.published_at.isoformat() if item.published_at else None,
                item.updated_at.isoformat() if item.updated_at else None,
                item.doi,
                item.arxiv_id,
                item.venue_or_blog,
                item.abstract_or_excerpt,
                item.body_text,
                json.dumps(item.tags),
                json.dumps(item.extra, default=str),
            ),
        )
        count += 1
    conn.commit()
    return count


def load_items(conn: sqlite3.Connection, *, limit: int = 500) -> list[SourceItem]:
    rows = conn.execute(
        "SELECT * FROM items ORDER BY COALESCE(published_at, updated_at, created_at) DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [_row_to_item(row) for row in rows]


def search(conn: sqlite3.Connection, query: str, *, limit: int = 10) -> list[SourceItem]:
    rows = conn.execute(
        """
        SELECT items.* FROM items_fts
        JOIN items ON items.rowid = items_fts.rowid
        WHERE items_fts MATCH ?
        ORDER BY bm25(items_fts)
        LIMIT ?
        """,
        (query, limit),
    ).fetchall()
    return [_row_to_item(row) for row in rows]


def update_scores(conn: sqlite3.Connection, ranked: Iterable[object]) -> None:
    for r in ranked:
        item = r.item
        conn.execute(
            """
            UPDATE items
            SET topic_scores_json=?, quality_signals_json=?, citation_signals_json=?,
                daily_score=?, deep_dive_score=?, updated_row_at=CURRENT_TIMESTAMP
            WHERE item_id=?
            """,
            (
                r.topic_scores.model_dump_json(),
                json.dumps(r.quality_signals, default=str),
                json.dumps(r.citation_signals, default=str),
                float(r.daily_score),
                float(r.deep_dive_score),
                item.item_id,
            ),
        )
    conn.commit()


def _row_to_item(row: sqlite3.Row) -> SourceItem:
    data = dict(row)
    authors = json.loads(data.pop("authors_json") or "[]")
    tags = json.loads(data.pop("tags_json") or "[]")
    extra = json.loads(data.pop("extra_json") or "{}")
    return SourceItem(
        item_id=data["item_id"],
        canonical_url=data["canonical_url"],
        source_kind=data["source_kind"],
        source_name=data["source_name"],
        source_tier=int(data["source_tier"]),
        title=data["title"],
        authors=authors,
        published_at=data["published_at"],
        updated_at=data["updated_at"],
        doi=data["doi"] or "",
        arxiv_id=data["arxiv_id"] or "",
        venue_or_blog=data["venue_or_blog"] or "",
        abstract_or_excerpt=data["abstract_or_excerpt"] or "",
        body_text=data["body_text"] or "",
        tags=tags,
        extra=extra,
    )
