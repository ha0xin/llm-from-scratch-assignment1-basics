# Tokenizer Experiments

(a) Compression ratio (bytes/token) on 10 sampled docs: TinyStories tokenizer on TinyStories = 4.248; OWT tokenizer on OWT = 4.682.

(b) OWT sample tokenized with TinyStories tokenizer: 3.261 bytes/token, which is lower than the OWT tokenizer (more tokens per byte), as expected for a smaller, less matched vocab.

(c) Throughput (tokenizer only, in-memory): 7601195.63 bytes/sec on 33.9 MB sample. Estimated time to tokenize 825GB (Pile) ≈ 32.4 hours (~1.3 days).

(d) uint16 is appropriate because vocab sizes are <= 32K, so all token IDs fit in 16 bits (0..65535), which halves storage vs int32.

Dataset tokenization not run in this script invocation. Use --tokenize-datasets to produce uint16 files.
