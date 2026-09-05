# Codexa C1 — Daily Progress Log
## 2026-09-05

### V9.1 Arabic Corpus — DuckDB

Today we completed and verified the first working V9.1 DuckDB Arabic corpus pipeline.

Source:
- Dataset: HuggingFaceFW/fineweb-2
- Config: arb_Arab
- Split: train
- Reader: DuckDB
- Development source shard: 000_00000.parquet
- Target test: 20,000 samples

### Environment
- Python 3.11.9
- PyTorch 2.6.0+cpu
- DuckDB 1.1.3
- datasets 2.19.0
- pyarrow 14.0.2
- NumPy 1.26.4
- CPU: Intel Core 2 Duo E8400
- CPU threads: 2

### Why DuckDB
PyArrow data-page decoding was unreliable on this old CPU/build even though Parquet metadata and raw ZSTD decoding worked.
DuckDB successfully decoded and streamed the same FineWeb-2 Parquet data.

Measured DuckDB reads:
- 1,000 rows: ~633.7 rows/s
- 10,000 rows: ~2,515 rows/s
- 50,000 rows: ~7,210 rows/s

### V9.1 20K Test Result
- Target: 20,000
- Accepted: 20,208
- Source rows processed: 621
- Quality rejects: 9,325
- Duplicate rejects: 351
- Decontamination rejects: 0
- Elapsed: 17.21 seconds
- Throughput: ~1,174.1 accepted samples/s

The accepted count exceeded the nominal target because processing respects complete source-row boundaries. The final source row contributed additional accepted samples.

### Output Shards
- shard 00000: 10,000 lines
- shard 00001: 10,000 lines
- shard 00002: 208 lines

### Checkpoint / Resume Decision
V9.1 checkpointing must always represent a completed FineWeb source-row boundary.

The previous implementation could checkpoint in the middle of a source record, which was unsafe because one FineWeb record can contain multiple lines.

New design:
- Advance source offset only after the complete source row is processed.
- Resume from source-row boundaries.
- Rebuild dedup state from existing shards.
- Automatic interruption/resume test will be used instead of relying on Ctrl+C.

### Quality Decisions
Current normalization baseline:
- NFKC
- remove Arabic Tatweel
- remove Arabic diacritics
- whitespace normalization

Current quality filters:
- minimum 30 characters
- maximum 20,000 characters
- Arabic ratio >= 0.50
- unique word ratio >= 0.30
- exact SHA256 deduplication
- tokenizer evaluation decontamination

These are a baseline, not the final quality ceiling. Future quality improvements must be measured rather than guessed.

### Performance Direction
The next optimization pass should improve throughput without weakening corpus quality.

Planned improvements:
- Replace repeated LIMIT/OFFSET Parquet reads with a single DuckDB streaming cursor.
- Reduce checkpoint I/O.
- Keep checkpoint safety at source-row boundaries.
- Optimize hot-path regex operations.
- Benchmark before/after.
- Preserve quality filtering, exact deduplication, and decontamination.

### Incident Log
An attempted PowerShell inline replacement introduced the literal string `` `r`n `` into Python and caused a SyntaxError.

The broken file was not accepted as a project milestone. The valid V9.1 file is restored from the verified backup before this checkpoint.

### Repository State
This checkpoint is intended to preserve today's Codexa C1 work before further performance optimization and automatic resume testing.

Next milestone:
1. Verify compile.
2. Automatic stop after first shard.
3. Verify checkpoint.
4. Restart.
5. Verify safe resume with no duplicate shard generation.
6. Remove test-stop mode.
7. Run performance benchmark.
8. Restore production target to 100,000.
