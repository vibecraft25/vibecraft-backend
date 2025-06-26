"""
MCP Data Upload and Conversion Server (CSV and SQLite only)

- Features:
  1. Upload general file via base64 and save (upload_file)
  2. Preview columns and sample rows from saved data (preview_data)
  3. Drop a specific column from CSV (drop_column_csv)
  4. Drop a specific column from SQLite DB (drop_column_sqlite)
  5. Save edited CSV as SQLite DB (save_sqlite)
  6. Delete previously saved files (delete_file)
  7. Reset all uploaded files (reset_all_files)
  8. List all uploaded files (list_uploaded_files)

- Data location:
  All saved files are stored under ./data_store/
"""

__author__ = "Se Hoon Kim(sehoon787@korea.ac.kr)"

# Standard imports
import os
import sys
import sqlite3
import base64

# Ensure project root on path for importing schemas
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Third-party imports
from fastmcp import FastMCP
import pandas as pd

# Custom imports
from schemas.data_upload_schemas import UploadData, FileUpload

# Initialize MCP instance
mcp = FastMCP("name=vibecraft_data_upload")

# Data directory setup
DATA_DIR = os.path.join(PROJECT_ROOT, "data_store")
os.makedirs(DATA_DIR, exist_ok=True)

# 1. Upload general file (base64)
@mcp.tool(
    name="upload_file",
    description="Save a file to the server (supports csv, sqlite).",
    tags={"upload", "file"}
)
def upload_file(data: FileUpload):
    try:
        file_path = os.path.join(DATA_DIR, data.filename)
        with open(file_path, "wb") as f:
            f.write(base64.b64decode(data.content_base64))
        return {"status": "success", "message": f"File '{data.filename}' saved.", "path": file_path}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# 2. Preview data
@mcp.tool(
    name="preview_data",
    description="Display columns and a sample row from CSV or SQLite.",
    tags={"preview", "data"}
)
def preview_data(filename: str):
    csv_path = os.path.join(DATA_DIR, f"{filename}.csv")
    sqlite_path = os.path.join(DATA_DIR, f"{filename}.sqlite")
    try:
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
        elif os.path.exists(sqlite_path):
            conn = sqlite3.connect(sqlite_path)
            df = pd.read_sql("SELECT * FROM records LIMIT 1", conn)
            conn.close()
        else:
            return {"status": "error", "message": "No file found."}
        return {"status": "success", "columns": list(df.columns), "sample_row": df.iloc[0].to_dict() if not df.empty else {}}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# 3. Drop column from CSV
@mcp.tool(
    name="drop_column_csv",
    description="Remove a specific column from a CSV file.",
    tags={"edit", "csv"}
)
def drop_column_csv(filename: str, column: str):
    path = os.path.join(DATA_DIR, f"{filename}.csv")
    if not os.path.exists(path):
        return {"status": "error", "message": "CSV file not found."}
    try:
        df = pd.read_csv(path)
        if column not in df.columns:
            return {"status": "error", "message": f"Column '{column}' not found."}
        df.drop(columns=[column], inplace=True)
        df.to_csv(path, index=False)
        return {"status": "success", "columns": list(df.columns)}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# 4. Drop column from SQLite
@mcp.tool(
    name="drop_column_sqlite",
    description="Remove a specific column from SQLite DB.",
    tags={"edit", "sqlite"}
)
def drop_column_sqlite(filename: str, column: str):
    path = os.path.join(DATA_DIR, f"{filename}.sqlite")
    if not os.path.exists(path):
        return {"status": "error", "message": "SQLite file not found."}
    try:
        conn = sqlite3.connect(path)
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(records)")
        cols = [row[1] for row in cursor.fetchall()]
        if column not in cols:
            return {"status": "error", "message": f"Column '{column}' not found."}
        remaining = [c for c in cols if c != column]
        rem_str = ", ".join(remaining)
        cursor.execute("ALTER TABLE records RENAME TO records_old")
        cursor.execute(f"CREATE TABLE records ({', '.join(f'{c} TEXT' for c in remaining)})")
        cursor.execute(f"INSERT INTO records ({rem_str}) SELECT {rem_str} FROM records_old")
        cursor.execute("DROP TABLE records_old")
        conn.commit()
        conn.close()
        return {"status": "success", "columns": remaining}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# 5. Save CSV to SQLite
@mcp.tool(
    name="save_sqlite",
    description="Save edited CSV records as a SQLite DB.",
    tags={"save", "sqlite"}
)
def save_sqlite(data: UploadData):
    if not data.records:
        return {"status": "error", "message": "No records to save."}
    sqlite_path = os.path.join(DATA_DIR, f"{data.filename}.sqlite")
    try:
        conn = sqlite3.connect(sqlite_path)
        cursor = conn.cursor()
        keys = data.records[0].keys()
        cols_def = ", ".join([f"{k} TEXT" for k in keys])
        cursor.execute(f"CREATE TABLE IF NOT EXISTS records ({cols_def})")
        ph = ", ".join(["?"] * len(keys))
        insert_q = f"INSERT INTO records VALUES ({ph})"
        for rec in data.records:
            cursor.execute(insert_q, tuple(str(rec[k]) for k in keys))
        conn.commit()
        conn.close()
        return {"status": "success", "path": sqlite_path}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# 6. Delete specific files
@mcp.tool(
    name="delete_file",
    description="Delete CSV or SQLite file.",
    tags={"delete", "file"}
)
def delete_file(filename: str):
    paths = [os.path.join(DATA_DIR, f"{filename}.{ext}") for ext in ("csv", "sqlite")]
    deleted = [p for p in paths if os.path.exists(p) and os.remove(p) is None]
    if not deleted:
        return {"status": "error", "message": "No file found to delete."}
    return {"status": "success", "deleted": deleted}

# 7. Reset all uploaded files
@mcp.tool(
    name="reset_all_files",
    description="Remove all files in data store.",
    tags={"reset", "file"}
)
def reset_all_files():
    try:
        names = []
        for f in os.listdir(DATA_DIR):
            p = os.path.join(DATA_DIR, f)
            if os.path.isfile(p):
                os.remove(p)
                names.append(f)
        return {"status": "success", "deleted": names}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# 8. List uploaded files
@mcp.resource("files://uploaded")
def list_uploaded_files():
    try:
        files = sorted(os.listdir(DATA_DIR))
        return {"status": "success", "files": files}
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8081, path="/mcp")
