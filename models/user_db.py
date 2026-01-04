"""
User Database Model
Simple SQLite database for user authentication
"""

import sqlite3
import hashlib
import os

DATABASE_PATH = 'users.db'

def init_db():
    """Initialize the database with users table"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            full_name TEXT,
            organization TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()
    print("✓ Database initialized")

def hash_password(password):
    """Hash password using SHA256"""
    return hashlib.sha256(password.encode()).hexdigest()

def create_user(username, email, password, full_name=None, organization=None):
    """Create a new user"""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()

        password_hash = hash_password(password)

        cursor.execute("""
            INSERT INTO users (username, email, password_hash, full_name, organization)
            VALUES (?, ?, ?, ?, ?)
        """, (username, email, password_hash, full_name, organization))

        conn.commit()
        conn.close()
        return True, "User created successfully"
    except sqlite3.IntegrityError:
        return False, "Username or email already exists"
    except Exception as e:
        return False, str(e)

def verify_user(username, password):
    """Verify user credentials"""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()

        password_hash = hash_password(password)

        cursor.execute("""
            SELECT id, username, email, full_name, organization 
            FROM users 
            WHERE username = ? AND password_hash = ?
        """, (username, password_hash))

        user = cursor.fetchone()

        if user:
            cursor.execute("""
                UPDATE users SET last_login = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (user[0],))
            conn.commit()

        conn.close()

        if user:
            return True, {
                'id': user[0],
                'username': user[1],
                'email': user[2],
                'full_name': user[3],
                'organization': user[4]
            }
        else:
            return False, None
    except Exception as e:
        return False, None

def get_user_by_username(username):
    """Get user by username"""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, username, email, full_name, organization, created_at
            FROM users WHERE username = ?
        """, (username,))

        user = cursor.fetchone()
        conn.close()

        if user:
            return {
                'id': user[0],
                'username': user[1],
                'email': user[2],
                'full_name': user[3],
                'organization': user[4],
                'created_at': user[5]
            }
        return None
    except Exception as e:
        return None

# Initialize database on import
if not os.path.exists(DATABASE_PATH):
    init_db()
