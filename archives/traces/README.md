# Compressed collection traces

One `tar.zst -19` archive per cell of `llm/traces/collection/` (STORAGE-POLICY
§7, Tier 2): 29 cells, ~3.7 GB of JSON → ~80 MB. Each archive extracts to exactly
the git-tracked files of its cell (round-trip verified with sha256 at creation).
The empty cell `e4b_r256_t10` has no archive.

The plain-JSON cells under `llm/traces/collection/` remain the working copy that
every reader script uses; these archives are a compact, portable snapshot of them.

Home-directory paths in `model_path` and error tails were scrubbed to `~` before
archiving (this repo is public).

```bash
sha256sum -c SHA256SUMS                                  # verify
zstd -dc armb_sota_r256_t10.tar.zst | tar -xf -          # extract one cell
```

Archives are built reproducibly (`--sort=name --mtime=@0 --owner=0 --group=0`),
so re-archiving an unchanged cell yields an identical file.
