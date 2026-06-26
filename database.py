import sqlite3
from datetime import datetime


DB_NAME = "blogs.db"


def get_connection():
    return sqlite3.connect(DB_NAME)



def create_table():

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS blogs(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        topic TEXT NOT NULL,
        content TEXT NOT NULL,
        created_at TEXT
    )
    """)

    conn.commit()
    conn.close()



def save_blog(title, topic, content):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO blogs(title, topic, content, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (
            title,
            topic,
            content,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
    )

    conn.commit()
    conn.close()



def get_all_blogs():

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT id, title, topic, content, created_at
        FROM blogs
        ORDER BY id DESC
        """
    )

    blogs = cursor.fetchall()

    conn.close()

    return blogs