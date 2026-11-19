# Upload verification record

Verification date: 2026-11-19

## Uploaded archive

- Path: `AAEC_reproducibility_public_preview_0.1.0-rc1_20261119_FIXED.zip`
- Size: 10,427,163 bytes
- SHA-256: `64881aabcc8bf7d5c34df6765201aa9581474bdde466500f12b1ca1f3855ebd5`
- Git blob SHA-1: `a180c3e989feed7c7f0464b357ab6738a1ed98b0`

The GitHub tree entry for the uploaded archive reported the same blob SHA-1
as the local file's Git object hash (computed as SHA-1 of
`blob <byte-length>\0<file-bytes>`). This establishes that the uploaded Git
object is byte-identical to the local archive.

## Archive inventory and checks

- Regular files: 822
- Directory entries: 72
- NPZ prediction archives: 145
- NPY index arrays: 2
- Recognized model-weight files: 0
- ZIP CRC/integrity test: PASS
- Bundled `scripts/verify_release.py`: PASS
- SHA-256 manifest entries: 821 data files plus the manifest's own header
  (the manifest does not hash itself)

The bundled verifier checked the file hashes in `SHA256SUMS.csv`, inspected
NPZ files with `allow_pickle=False`, required prediction-oriented arrays,
checked that NPY files are one-dimensional integer index vectors, and
rejected common model-weight extensions.

## Remaining publication checks

This record verifies archive upload and internal package integrity. It does
not establish coauthor approval, third-party dataset rights beyond the
included notices, complete historical experiment provenance, GitHub secret
scan clearance, or a DOI-backed archival release. Those remain separate
publication checks.
