import sqlite3

DATABASE = "plant_predictions.db"


def get_connection():
    connection = sqlite3.connect(DATABASE)
    connection.row_factory = sqlite3.Row
    return connection


def create_database():
    connection = get_connection()

    connection.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_name TEXT NOT NULL,
            disease TEXT NOT NULL,
            confidence REAL NOT NULL,
            status TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    connection.commit()
    connection.close()


def add_prediction(image_name, disease, confidence, status):
    connection = get_connection()

    connection.execute("""
        INSERT INTO predictions
        (image_name, disease, confidence, status)
        VALUES (?, ?, ?, ?)
    """, (
        image_name,
        disease,
        confidence,
        status
    ))

    connection.commit()
    connection.close()


def get_predictions():
    connection = get_connection()

    predictions = connection.execute("""
        SELECT *
        FROM predictions
        ORDER BY id DESC
    """).fetchall()

    connection.close()

    return predictions


def get_statistics():
    connection = get_connection()

    total = connection.execute(
        "SELECT COUNT(*) FROM predictions"
    ).fetchone()[0]

    healthy = connection.execute(
        "SELECT COUNT(*) FROM predictions WHERE status LIKE '%Healthy%'"
    ).fetchone()[0]

    diseased = total - healthy

    connection.close()

    return total, healthy, diseased


if __name__ == "__main__":
    create_database()
    print("Database created successfully!")