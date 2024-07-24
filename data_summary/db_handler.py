import sqlite3
import csv
from typing import Optional
import pandas as pd


class DBConnection:
#TODO merge into one class if pragma keys won't be used
    def __init__(self, db_name: str):
        self.db_name = db_name
        self.conn = sqlite3.connect(self.db_name)

    def close(self):
        self.conn.close()


class DBHandler:

    def __init__(self, db_name: str):
        self.db_name = db_name
        self.conn = sqlite3.connect(self.db_name)
        self.cursor = self.conn.cursor()

    def table_exists(self, table_name: str) -> bool:
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
        return self.cursor.fetchone() is not None

    def create_table_from_csv(self, csv_file: str, table_name: str, delimiter: Optional[str] = ',') -> None:
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader)
            columns = delimiter.join([f'{header} TEXT' for header in headers])

            create_table_query = f'CREATE TABLE IF NOT EXISTS {table_name} ({columns})'
            self.cursor.execute(create_table_query)

            insert_query = f'INSERT INTO {table_name} ({", ".join(headers)}) VALUES ({", ".join(["?" for _ in headers])})'
            for row in reader:
                self.cursor.execute(insert_query, row)
            self.conn.commit()

    def _load_query(self, file_path: str) -> str:
        with open(file_path, 'r') as file:
            return file.read()

    def execute_query(self, query_file: str) -> pd.DataFrame:
        query = self._load_query(file_path=query_file)
        return pd.read_sql_query(query, self.conn)

    def close(self):
        self.conn.close()
