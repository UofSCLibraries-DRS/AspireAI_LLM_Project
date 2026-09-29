# 1. LLM-Generated Keywords

Ask the LLM to generate keywords to search for to help answer the query. Make this query very simple.

Parse out these keywords from the LLM and use it with the pg database using BM25 to search for relavent documents.


# 2. Hybrid Approach

Combine both the aforementioned keyword based search with vector embeddings. I saw an algorithm called RRF for this.

# 3.  HyDE (Answer)

Ask the model to generate a hypothetical anwer to the query and perform vector search with that.

# 4.  HyDE (Document)

Ask the model to generate a hypothetical document that would answer the query and perform vector search with that.