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
scp -i <path_to_ssh_key> \
  demo/backend/db/data/embeddings.csv \
  <user>@<ip>:<path_to_repo>/demo/backend/db/data/embeddings.csv
```


## Database Initialization

Create a new database with:

```sql
CREATE DATABASE lighthouse_rag;
```

Confirm that the new database exists with:

```
\list
```

From the terminal, add the database schema with:

```bash
psql -U postgres -d lighthouse_rag -f schema.sql
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

Connection settings such as `PGHOST`, `PGPORT`, `PGUSER`, and `PGPASSWORD` are
honored by `psql`; `--host`, `--port`, and `--user` may also be supplied.
