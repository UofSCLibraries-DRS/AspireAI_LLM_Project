# Database Docs

### On the EC2 instance

The database is hosted locally on the EC2 instance. Connect to it with:

```bash
sudo -u postgres psql
```

## Creating embeddings

Run the generator from `demo/backend`:

```bash
uv run python db/create_embeddings.py
```

These embeddings are ignored by git so you will have to manually transfer them from the local machine to the hosting machine:

```bash
ssh -i <path_to_ssh_key> <user>@<ip> \
  'mkdir -p <path_to_repo>/demo/backend/db/data'

scp -i <path_to_ssh_key> \
  demo/backend/db/data/embeddings.csv \
  <user>@<ip>:<path_to_repo>/demo/backend/db/data/embeddings.csv
```


## Database Initialization

Create the database, then create a PostgreSQL login matching the Linux account
that will run the loader. The following uses the EC2 account `jaaydin`; replace
it if yours differs.

```bash
sudo -u postgres createdb lighthouse_rag
sudo -u postgres createuser --login jaaydin
sudo -u postgres psql -d lighthouse_rag -c \
  'ALTER DATABASE lighthouse_rag OWNER TO jaaydin;'
```

If the role already exists, `createuser` will report that fact; continue with
the ownership command. Confirm that the database exists with:

```bash
sudo -u postgres psql -l
```

Install `pgvector` and `pg_textsearch` in the database once. This step requires
the PostgreSQL superuser; the normal loader does not. `pg_textsearch` must also
be installed on the server, listed in `shared_preload_libraries`, and loaded by
restarting PostgreSQL before this command is run.

```bash
sudo -u postgres psql -d lighthouse_rag -v ON_ERROR_STOP=1 \
  -c 'CREATE EXTENSION IF NOT EXISTS vector;' \
  -c 'CREATE EXTENSION IF NOT EXISTS pg_textsearch;'
```


## Loading the generated embeddings

From `demo/backend`, load `db/data/embeddings.csv` into the local database:

```bash
python db/load_embeddings.py
```

The loader detects the embedding dimension before it imports and uses the
PostgreSQL client `psql` for a streaming CSV import. PostgreSQL validates every
remaining vector as it loads. The generated embeddings are 768-dimensional. To
deliberately replace rows already in `documents`, use:

```bash
python db/load_embeddings.py --replace
```

The loader creates the `documents` table when needed, so applying `schema.sql`
manually is not required. Run it as the matching Linux/PostgreSQL user (for
example, `jaaydin`), not with `sudo`.

After loading the data, create the HNSW cosine indexes and the generated
`search_text` column with its BM25 index. This is a one-time operation and may
take a while on the full dataset:

```bash
psql -d lighthouse_rag -f db/schema.sql
```

Connection settings such as `PGHOST`, `PGPORT`, `PGUSER`, and `PGPASSWORD` are
honored by `psql`; `--host`, `--port`, and `--user` may also be supplied.
