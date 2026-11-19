# Configuration notes

Set the following environment variables before using the supplied configurations:

```bash
export AAEC_DATA_ROOT=/path/to/datasets
export AAEC_PRETRAINED_ROOT=/path/to/pretrained/checkpoints
```

The configurations preserve the manuscript-level optimizer, schedule, input size, alpha grid, and PEFT settings. Dataset split manifests have not yet been recovered for all datasets; see `docs/RELEASE_BLOCKERS.md`.
