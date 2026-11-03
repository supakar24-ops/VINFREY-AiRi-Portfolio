"""Explicit local notes in SQLite; no conversation logging or extra dependencies."""

import hashlib
import re
import sqlite3
import unicodedata
from contextlib import closing
from pathlib import Path

DEFAULT_DB_PATH = Path(__file__).resolve().parent / "local_data" / "memory.sqlite3"
MAX_MEMORY_LENGTH = 2000


def content_hash(content):
    """Identify a note's content without keeping the text itself."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _checked(content):
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Memory must not be empty.")
    content = content.strip()
    if len(content) > MAX_MEMORY_LENGTH:
        raise ValueError(f"Memory must be at most {MAX_MEMORY_LENGTH} characters.")
    return content


def normalize_memory(text):
    """Comparable form for duplicate checks: NFKC, casefold, collapsed whitespace."""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


class MemoryStore:
    def __init__(self, path=DEFAULT_DB_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)

    def save(self, content):
        """Call only when the user explicitly asks to remember a note."""
        content = _checked(content)
        with closing(sqlite3.connect(self.path)) as connection, connection:
            cursor = connection.execute(
                "INSERT INTO memories (content) VALUES (?)", (content,)
            )
            return cursor.lastrowid

    def find_duplicate(self, content):
        """Return the id of a saved note equal after normalization, or None."""
        key = normalize_memory(content)
        for note in self.list_memories():
            if normalize_memory(note["content"]) == key:
                return note["id"]
        return None

    def save_if_absent(self, content):
        """Save only if no equal note exists; check and insert under one write lock.

        Returns the new id, or None when a duplicate was already saved.
        """
        content = _checked(content)
        key = normalize_memory(content)
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute("SELECT content FROM memories").fetchall()
            if any(normalize_memory(row[0]) == key for row in rows):
                return None
            return connection.execute(
                "INSERT INTO memories (content) VALUES (?)", (content,)
            ).lastrowid

    def list_memories(self):
        with closing(sqlite3.connect(self.path)) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                "SELECT id, content, created_at FROM memories ORDER BY id"
            ).fetchall()
        return [dict(row) for row in rows]

    def get(self, note_id):
        with closing(sqlite3.connect(self.path)) as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                "SELECT id, content, created_at FROM memories WHERE id = ?", (note_id,)
            ).fetchone()
        return dict(row) if row else None

    def delete_if_unchanged(self, note_id, content_sha256, created_at):
        """Delete one note only if it is still the exact note that was approved.

        The id alone is not enough: without AUTOINCREMENT, SQLite can reuse it.
        Content hash plus created_at guard against that, in one transaction.
        """
        with closing(sqlite3.connect(self.path)) as connection, connection:
            # Take the write lock before reading so nothing changes between check and delete.
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT content, created_at FROM memories WHERE id = ?", (note_id,)
            ).fetchone()
            if row is None or content_hash(row[0]) != content_sha256 or row[1] != created_at:
                return False
            connection.execute("DELETE FROM memories WHERE id = ?", (note_id,))
            return True

    def relevant(self, query, limit=5):
        """Simple case-insensitive keyword matching, including Thai substrings.

        This is deliberately not semantic search. Newer notes break score ties.
        """
        terms = set(re.findall(r"[\w\u0e00-\u0e7f]+", query.casefold()))
        ranked = []
        for note in self.list_memories():
            content = note["content"].casefold()
            score = sum(term in content for term in terms)
            if score:
                ranked.append((score, note["id"], note))
        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [note for _, _, note in ranked[:limit]]
