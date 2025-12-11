from typing import List, Any, Optional, Dict
import sqlite3
import time

from models import RetrievalResult


def boot_inmemory_sqlite() -> sqlite3.Connection:
    """
    Create an in-memory SQLite DB with a single 'work' table
    that stands in for structured project metadata.
    """
    conn = sqlite3.connect(":memory:")
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE work (
            id INTEGER PRIMARY KEY,
            name TEXT,
            location TEXT,
            client TEXT,
            year INTEGER,
            status TEXT,
            program TEXT,
            partner TEXT,
            associate TEXT,
            team TEXT
        )
        """
    )

    demo_rows = [
        (
            1,
            "2025 FW Miu Miu Show",
            "Paris, France",
            "Miu Miu",
            2025,
            "Completed",
            "Scenography",
            "Rem Koolhaas",
            "Giulio Margheri",
            "Luisa Carvalho, Raffaele Guercia",
        ),
        (
            2,
            "Busan Slope Housing",
            "",
            "Busan Architecture Festival",
            2025,
            "Commissioned Study",
            "Masterplan",
            "Chris van Dujn",
            "John Thurtle",
            "Jeremy Chow, Freddy Maggiorani, Xaveer Roodbeen, Felicia Gambino, Suhin Park",
        ),
        (
            3,
            "JOMOO Headquarters",
            "Xiamen, China",
            "Jomoo",
            2025,
            "Completed",
            "Office",
            "Chris van Dujin",
            "",
            "Mark Bavoso, Slava Savova, Sebastian Schulte, Pu Hsien Chan, Alan Lau, Chen Lu, Slava Savova, Sebastian Schulte, Ricky Suen, Gabriele Ubareviciute, Yue Wu, Adisak Yavilas, Cecilia Lei, Chen Lu, Kevin Mak, Ricky Suen, Connor Sullivan, Gabriele Ubareviciute, Chen Lu, Lingxiao Zhang",
        ),
        (
            4,
            "Casa Wabi Mushroom Pavilion",
            "Puerto Escondido, Mexico",
            "Fundación Casa Wabi",
            2024,
            "Construction",
            "Mixed Use",
            "Shohei Shigematsu",
            "",
            "Caroline Corbett, Shary Tawil, Francesco Rosati, Dylan Wei",
        ),
        (
            5,
            "AIR - Circular Campus and Cooking Club",
            "",
            "Potato Head",
            2024,
            "Completed",
            "Restaurant / Bar",
            "David Gianotten",
            "Shinji Takagi",
            "Marina Bonet, Matteo Fontana, Helena Daher Gomes, Raffaele Guercia, Marc Heumer, Alisa Kutsenko, Maria Aller Rey, Arthur Wong, Suet Ying Yuen",
        ),
        (
            6,
            "Brooklyn Academy of Music",
            "New York City, USA",
            "Brooklyn Academy of Music Local Development Corporation",
            2000,
            "Commissioned Study",
            "Mixed Use",
            "Rem Koolhaas",
            "",
            "Dan Wood, Eric Chang, Matthias Hollwich, Thorsten Kiefer, Casey Mack, Will Prince, Julien De Smedt, Sybille Wälty",
        ),
    ]

    cur.executemany(
        """
        INSERT INTO work(id, name, location, client, year, status, program, partner, associate, team)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        demo_rows,
    )
    conn.commit()
    return conn


class SQLRetriever:
    """
    Minimal SQL retrieval into dict rows.
    Intentional: only SELECT, fixed schema, parameterized WHERE.
    """

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def retrieve(self, language: Optional[str] = None, limit: int = 10) -> RetrievalResult:
        t0 = time.time()
        sql = "SELECT * FROM work"
        params: List[Any] = []

        clauses: List[str] = []
        if language:
            # Example heuristic: filter by language name appearing in project name or client
            clauses.append("(LOWER(name) LIKE ? OR LOWER(client) LIKE ?)")
            like_val = f"%{language}%"
            params.extend([like_val, like_val])

        if clauses:
            sql += " WHERE " + " AND ".join(clauses)

        sql += " LIMIT ?"
        params.append(limit)

        cur = self.conn.cursor()
        cur.execute(sql, params)
        cols = [d[0] for d in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]

        diagnostics: Dict[str, Any] = {
            "latency_ms": (time.time() - t0) * 1000,
            "sql": sql,
            "params": params,
        }
        return RetrievalResult(sql_rows=rows, passages=None, diagnostics=diagnostics)
