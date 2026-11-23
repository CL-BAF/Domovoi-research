# NAS dataset inventory · 24 November 2026

This was a read-only source audit on an SMB share. Windows could read the UNC path and mapped Z: drive. Active packing and training used local Windows NVMe. WSL was unavailable. This report omits private host paths and records split-relative names and hashes in the [machine-readable inventory](data/NAS-INVENTORY-2026-11-24.json).

| Corpus | Files | Bytes on NAS | Decoded records | Blank records | Null records |
|---|---:|---:|---:|---:|---:|
| TinyStories | 2 | 1,943,728,838 | 2,141,709 delimiter records | 230 | 0 |
| WikiText-103 raw | 4 | 315,466,397 | 1,809,468 Parquet rows | 639,087 | 0 |
| FineWeb-Edu 10BT | 14 | 28,518,193,415 | 9,672,101 Parquet rows | 0 | 0 |

All 20 expected files were present and matched the recorded Hugging Face download SHA-256/ETag. No incomplete or temporary download files were found. The text reader used strict UTF-8. The audit read the Parquet schema and every row group of the text column, validated Arrow arrays, and enabled Parquet page checksum verification where checksums exist. It did not decode every non-text column. WikiText blank rows are layout separators, counted in raw audit and skipped by the article reader. Content and licensing restrictions are not inferred from file validity.

- **TinyStories:** revision `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`, CDLA-Sharing-1.0 as listed by the dataset provider. This is synthetic narrative text, useful for controlled training calibration but narrow in domain. [Pinned dataset](https://huggingface.co/datasets/roneneldan/TinyStories/tree/f54c09fd23315a6f9c86f9dc80f725de7d8f9c64)
- **WikiText-103 raw:** revision `b08601e04326c79dfdd32d625aee71d232d685c3`. The dataset card metadata lists CC BY-SA 3.0 and GFDL, while the card's prose references CC BY-SA 4.0. Preserve attribution and check the original terms before redistribution. This report publishes only derived counts and hashes. [Dataset licensing section](https://huggingface.co/datasets/Salesforce/wikitext#licensing-information)
- **FineWeb-Edu 10BT sample:** revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`, ODC-BY 1.0 and subject to Common Crawl terms. Only a one-shard preview was packed. [Dataset terms](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu#licensing-information)

## Prepared versions

| Version | Train tokens | Validation tokens | Test tokens | Train documents |
|---|---:|---:|---:|---:|
| TinyStories v1 | 110,052,057 | 4,805,339 | 2,018,159 | 487,424 |
| WikiText-103 v1 | 152,441,047 | 322,861 | 368,001 | 28,998 |
| FineWeb-Edu preview v1 | 5,044,464 | 371,484 | 413,911 | 3,584 |

The [packing manifest](data/PACKED-MANIFEST-2026-11-24.json) contains split, boundary and tokenizer hashes. All three versions passed checks of token ID range, exact offset count, strictly increasing document ends, and presence of EOS at every document end.

TinyStories retains the official validation set. A deterministic normalized-document hash designates a 1% holdout from the first one million official-train records, with no second validation set. The streaming stage rejected 71 invalid records and 74,140 normalized exact duplicates across this input and the earlier-ingested official validation documents. Train packing stopped at the first token batch boundary beyond 110 million. The reserved test split has not been used for calibration. WikiText retained all official split assignments and grouped text rows by top-level headings within each shard. FineWeb-Edu preview read 30,000 document records from the first shard, indexed all three deterministic partitions, and packed a little over five million training tokens.

Tokenizers are trained once per prepared corpus on a bounded sample of eligible train documents. Every arm in a comparison shares one tokenizer, including its exact hash. The tokenizer uses a byte alphabet, a 4,096-token vocabulary and a distinct EOS ID. Normalized exact matching does not detect all templated or near-duplicate stories; contamination against unrelated benchmarks was not assessed. Benchmark files were not included in ingestion.

One failed preparation attempt exposed Rust tokenizer worker threads crossing SQLite's thread boundary. The staged document journal was intact. A bounded train-only file spool replaced the SQLite iterator at the tokenizer boundary, and preparation resumed from the existing journal. Train packing retains a separate per-split journal so previously committed batches survive interruptions. [Implementation workflow](https://github.com/CL-BAF/Domovoi-Labs/blob/f5fac7a34a211adf93e0dc8ea576c8d973ef87ac/docs/REAL_DATA_WORKFLOW.md)
