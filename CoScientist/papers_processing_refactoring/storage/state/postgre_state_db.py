# from datetime import datetime, timezone
# from typing import Any, Dict, List, Optional
#
# import psycopg2
# from psycopg2.extras import RealDictCursor
#
# from .base import BaseStateManager
#
#
# class PostgreSQLStateManager(BaseStateManager):
#     def __init__(self, dsn: str):
#         """
#         dsn: строка подключения типа "host=localhost dbname=test user=postgres password=secret"
#         или URL "postgresql://user:pass@host:port/dbname"
#         """
#         self.conn = psycopg2.connect(dsn)
#         self._init_schema()
#
#     def _init_schema(self):
#         with self.conn.cursor() as cur:
#             cur.execute("""
#             CREATE TABLE IF NOT EXISTS article_steps (
#                 article_id TEXT NOT NULL,
#                 step_name TEXT NOT NULL,
#                 status TEXT NOT NULL,
#                 updated_at TIMESTAMPTZ NOT NULL,
#                 error TEXT,
#                 PRIMARY KEY (article_id, step_name)
#             )
#             """)
#         self.conn.commit()
#
#     def get_status(self, article_id: str, step: str) -> Optional[str]:
#         with self.conn.cursor() as cur:
#             cur.execute(
#                 "SELECT status FROM article_steps WHERE article_id=%s AND step_name=%s",
#                 (article_id, step),
#             )
#             row = cur.fetchone()
#             return row[0] if row else None
#
#     def set_status(self, article_id: str, step: str, status: str, error: str | None = None):
#         now = datetime.now(timezone.utc)
#         with self.conn.cursor() as cur:
#             cur.execute("""
#             INSERT INTO article_steps(article_id, step_name, status, updated_at, error)
#             VALUES (%s, %s, %s, %s, %s)
#             ON CONFLICT(article_id, step_name)
#             DO UPDATE SET
#                 status=EXCLUDED.status,
#                 updated_at=EXCLUDED.updated_at,
#                 error=EXCLUDED.error
#             """, (article_id, step, status, now, error))
#         self.conn.commit()
#
#     def list_states(self, article_id=None, status=None, step=None) -> List[Dict[str, Any]]:
#         query = "SELECT * FROM article_steps"
#         params = []
#         conditions = []
#
#         if article_id:
#             conditions.append("article_id = %s")
#             params.append(article_id)
#         if status:
#             conditions.append("status = %s")
#             params.append(status)
#         if step:
#             conditions.append("step_name = %s")
#             params.append(step)
#
#         if conditions:
#             query += " WHERE " + " AND ".join(conditions)
#
#         # Используем RealDictCursor для возврата словарей
#         with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
#             cur.execute(query, params)
#             return [dict(row) for row in cur.fetchall()]
#
#     def clear_data(self, article_id: Optional[str] = None):
#         with self.conn.cursor() as cur:
#             if article_id:
#                 cur.execute("DELETE FROM article_steps WHERE article_id = %s", (article_id,))
#             else:
#                 cur.execute("DELETE FROM article_steps")
#         self.conn.commit()
#
#     def reset_running_states(self, message: str = "Interrupted by system restart"):
#         now = datetime.now(timezone.utc)
#         with self.conn.cursor() as cur:
#             cur.execute("""
#                 UPDATE article_steps
#                 SET status = 'failed', error = %s, updated_at = %s
#                 WHERE status = 'running'
#             """, (message, now))
#         self.conn.commit()
#
#     def close(self):
#         self.conn.close()