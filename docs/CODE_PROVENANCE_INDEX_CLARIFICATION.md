# Clarification of the archived code-provenance index

The public preview ZIP contains `code/CODE_PROVENANCE.csv`. Its first column is
currently named `public_path`, but that label is stale and misleading.

The 35 paths listed in that column refer to recovered source files in internal
candidate archives. **Those 35 runtime-source files are not distributed in
the public preview ZIP**; none of the listed paths exists in the public archive.
The `public_file_sha256` column records hashes for those recovered candidate
files, not files shipped in the public package.

The authoritative public distribution scope is described in the archive's
`docs/CODE_PROVENANCE.md` and `LICENSE_SCOPE.md`:

- `src/aaec/selection.py`: portable reference implementation of the formal
  five-point joint Bicubic/Bilinear validation selector;
- `tests/test_selection.py` and `scripts/verify_release.py`: selector and
  release verification tests;
- `code/figure_generation/`: figure-generation scripts and helpers.

Recovered `tasfd` foundation/runtime sources and C60 training/evaluation
runner snapshots are excluded because complete upstream copyright and license
provenance has not been established. This note corrects the interpretation of
the legacy CSV; it does not add or imply public availability of those files.
