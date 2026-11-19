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
- [ ] Upload the full preview directory to this GitHub repository.
- [ ] Confirm all coauthors approve public distribution of the included derived data.
- [ ] Run GitHub secret scanning after upload.
- [ ] Create a versioned release after the authors' final review; then archive that exact release to Zenodo if a DOI is desired.

## Deferred items

Exact full split manifests, additional ImageNet-100 seed records, expanded
historical run-to-source traceability, and file-level third-party source
license review are deferred and described in
`docs/OPEN_ITEMS_AND_LIMITATIONS.md`. They must not be described as completed.
