# Repository security review — 2026-11-20

## GitHub controls

The repository owner confirmed on 2026-11-20 that the GitHub Secret scanning
alerts page showed no warnings. Screenshots provided in the project
conversation show:

- Secret Protection: enabled;
- Push protection: On.

This records the author's inspection of the repository's current GitHub alert
state. It is not an independent assertion that every historical commit,
branch, or unsupported secret format has been exhaustively scanned.

## Local archive scan

The uploaded public preview ZIP
`AAEC_reproducibility_public_preview_0.1.0-rc1_20261119_FIXED.zip` was
scanned locally. SHA-256:
`64881aabcc8bf7d5c34df6765201aa9581474bdde466500f12b1ca1f3855ebd5`.

The scan covered 822 regular archive files, including 652 text-like files,
and searched for common AWS/GitHub/Slack/Google/OpenAI token patterns, PEM
private-key markers, generic credential assignments, embedded URL
credentials, and suspicious credential filenames. No matching patterns or
suspicious filenames were detected.

This is heuristic static scanning, not a substitute for GitHub's scanning
service or a guarantee against unknown credential formats. If a real
credential is later identified, it must be revoked or rotated; deleting the
visible file alone is insufficient.
