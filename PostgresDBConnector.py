import os
import psycopg2
from psycopg2 import pool
from dotenv import load_dotenv
from contextlib import contextmanager

# Load environment variables
load_dotenv()


class PostgresDBConnector:
    """PostgreSQL database connector with connection pooling and proper resource management."""

    def __init__(self):
        self.connection_params = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': int(os.getenv('DB_PORT', '5432')),
            'database': os.getenv('DB_NAME', 'example_db'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', 'password'),
        }
        self.connection_pool = None
        self._initialize_pool()

    def _initialize_pool(self):
        """Initialize the connection pool."""
        try:
            self.connection_pool = psycopg2.pool.SimpleConnectionPool(
                minconn=1,
                maxconn=10,
                **self.connection_params
            )
            print(f"✓ Connection pool created successfully to {self.connection_params['host']}:{self.connection_params['port']}")
        except psycopg2.Error as e:
            print(f"✗ Error creating connection pool: {e}")
            raise

    @contextmanager
    def get_connection(self):
        """Context manager for getting a connection from the pool."""
        connection = None
        try:
            connection = self.connection_pool.getconn()
            yield connection
        except psycopg2.Error as e:
            print(f"✗ Database error: {e}")
            if connection:
                connection.rollback()
            raise
        finally:
            if connection:
                self.connection_pool.putconn(connection)

    @contextmanager
    def get_cursor(self, commit=False):
        """Context manager for getting a cursor with automatic commit/rollback."""
        with self.get_connection() as connection:
            cursor = connection.cursor()
            try:
                yield cursor
                if commit:
                    connection.commit()
            except psycopg2.Error as e:
                connection.rollback()
                print(f"✗ Error executing query: {e}")
                raise
            finally:
                cursor.close()

    def execute_query(self, query, params=None, fetch=True):
        """Execute a query and optionally fetch results."""
        with self.get_cursor() as cursor:
            cursor.execute(query, params)
            if fetch:
                return cursor.fetchall()
            return None

    def execute_commit(self, query, params=None):
        """Execute a query with commit (for INSERT, UPDATE, DELETE)."""
        with self.get_cursor(commit=True) as cursor:
            cursor.execute(query, params)
            return cursor.rowcount

    def close_pool(self):
        """Close all connections in the pool."""
        if self.connection_pool:
            self.connection_pool.closeall()
            print("✓ Connection pool closed")


# Example usage
if __name__ == "__main__":
    try:
        # Initialize connector
        db = PostgresDBConnector()

        # Execute a simple query
        result = db.execute_query("SELECT * FROM items;")
        print("Data from Database:", result)

        # Close the pool when done
        db.close_pool()

    except psycopg2.Error as e:
        print(f"✗ Database connection failed: {e}")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")