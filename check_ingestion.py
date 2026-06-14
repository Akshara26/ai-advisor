# check_ingestion.py
import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

try:
    import streamlit as st
    db_url = st.secrets["SUPABASE_DB_URL"]
except Exception:
    db_url = os.getenv("SUPABASE_DB_URL")

db_url = db_url.replace("postgres://", "postgresql://", 1)

conn = psycopg2.connect(db_url)
cur = conn.cursor()

# 1. Total chunk count
cur.execute("SELECT COUNT(*) FROM data_umn_handbook")
total = cur.fetchone()[0]
print(f"\nTotal chunks: {total}")

# 2. Breakdown by source domain
cur.execute("""
    SELECT metadata_->>'source' as source, COUNT(*) as chunks
    FROM data_umn_handbook
    GROUP BY source
    ORDER BY chunks DESC
""")
print("\nChunks by source:")
for row in cur.fetchall():
    print(f"  {row[1]:4d}  {row[0]}")

# 3. Breakdown by category
cur.execute("""
    SELECT metadata_->>'category' as category, COUNT(*) as chunks
    FROM data_umn_handbook
    WHERE metadata_->>'category' IS NOT NULL
    GROUP BY category
    ORDER BY chunks DESC
""")
print("\nChunks by category:")
for row in cur.fetchall():
    print(f"  {row[1]:4d}  {row[0]}")

# Replace the last two queries with:

# 4. Most recently ingested (fallback to url list if no ingested_at)
cur.execute("""
    SELECT DISTINCT metadata_->>'url'
    FROM data_umn_handbook
    WHERE metadata_->>'ingested_at' IS NOT NULL
    ORDER BY metadata_->>'url'
    LIMIT 10
""")
rows = cur.fetchall()
if rows:
    print("\nURLs with ingested_at set:")
    for row in rows:
        print(f"  {row[0]}")
else:
    print("\nNo ingested_at timestamps found (older ingestion runs didn't set this field).")

# 5. Short chunks
cur.execute("""
    SELECT metadata_->>'url', length(text) as text_len
    FROM data_umn_handbook
    WHERE length(text) < 100
    ORDER BY text_len ASC
    LIMIT 10
""")
rows = cur.fetchall()
if rows:
    print(f"\nSuspiciously short chunks (< 100 chars):")
    for row in rows:
        print(f"  {row[1]} chars  {row[0]}")
else:
    print("\nNo suspiciously short chunks found.")

cur.close()
conn.close()