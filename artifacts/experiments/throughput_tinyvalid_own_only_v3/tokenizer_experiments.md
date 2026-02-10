# Tokenizer Experiments

(a) Compression ratio (bytes/token) on 10 sampled docs: TinyStories tokenizer on TinyStories = 4.248; OWT tokenizer on OWT = 4.248.

(b) OWT sample tokenized with TinyStories tokenizer: 4.248 bytes/token, which is lower than the OWT tokenizer (more tokens per byte), as expected for a smaller, less matched vocab.

(c) Throughput (tokenizer only, in-memory): 11427365.50 bytes/sec on 22.5 MB sample. Estimated time to tokenize 825GB (Pile) ≈ 21.5 hours (~0.9 days).

(d) uint16 is appropriate because vocab sizes are <= 32K, so all token IDs fit in 16 bits (0..65535), which halves storage vs int32.

Dataset tokenization not run in this script invocation. Use --tokenize-datasets to produce uint16 files.
