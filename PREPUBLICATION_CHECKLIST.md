# Public preview checklist

- [x] Separate new repository target; do not modify the historical `AAEC` repository.
- [x] MIT license for original AAEC selector/figure software.
- [x] CC BY 4.0 notice for author-created derived records and manifests.
- [x] Explicit license-scope exclusions for third-party materials.
- [x] Evidence hierarchy and known provenance limitations documented.
- [x] Runtime source snapshots with unresolved third-party license attribution excluded.
- [x] Model weights/checkpoints and raw benchmark images excluded.
- [x] SHA-256 manifest generated and release verification passed locally.
- [x] Portable selector unit tests passed locally.
- [x] Public preview ZIP uploaded to the new GitHub repository.
- [x] Uploaded ZIP byte identity checked against the local archive using the Git blob SHA-1.
- [x] ZIP CRC check and the bundled `scripts/verify_release.py` passed locally.
- [x] Confirm all coauthors approve public distribution of the included derived data (author confirmation received on 2026-11-20).
- [ ] Run GitHub secret scanning and manually review its results.
- [ ] Create a versioned GitHub release after the authors' final review; then archive that exact release to Zenodo if a DOI is desired.

## Deferred items

Exact full split manifests, additional ImageNet-100 seed records, expanded
historical run-to-source traceability, and file-level third-party source
license review are deferred and described in
`docs/OPEN_ITEMS_AND_LIMITATIONS.md` inside the ZIP archive. They must not be
described as completed.
