# Local archive security scan — 2026-11-20

## Scope

Scanned the uploaded public preview archive:

- File: `AAEC_reproducibility_public_preview_0.1.0-rc1_20261119_FIXED.zip`
- SHA-256: `64881aabcc8bf7d5c34df6765201aa9581474bdde466500f12b1ca1f3855ebd5`
- ZIP entries: 894 (822 regular files and 72 directory entries)
- Text-like files scanned: 652

## Checks performed

A local heuristic scan searched text files and raw archive-member bytes for
common provider credential patterns and private-key markers, including:

- AWS access-key identifiers;
- GitHub personal-access-token patterns;
- Slack token patterns;
- Google API-key patterns;
- OpenAI-style secret-key prefixes;
- PEM private-key headers;
- generic API-key, password, token, client-secret, and private-key assignments;
- URLs containing embedded username/password credentials;
- common environment-variable credential assignments;
- likely internal filesystem paths, SSH/SFTP URLs, and server endpoint settings.

## Result

**No matching credential or private-key patterns were detected** by these
local checks. No suspicious credential-bearing filenames were found.

The ZIP integrity test passed. The bundled release verifier passed, including
its checks of the SHA-256 manifest, NPZ prediction-array structure, NPY index
arrays, and absence of model-weight files. The portable selector tests passed
(3 tests). The figure-data validator passed, and the figure-generation script
generated 10 PDF and 10 PNG figures.

## Important limitation

This is a local heuristic scan of the uploaded ZIP snapshot. It is not a
replacement for GitHub Secret Scanning, does not inspect every possible
unknown credential format, and does not scan the repository's complete Git
history or other branches. GitHub Secret Scanning should still be reviewed
from the repository's Security and quality interface. If a real credential
is ever detected, revoke or rotate it immediately; deleting the visible file
alone does not invalidate an exposed credential.
