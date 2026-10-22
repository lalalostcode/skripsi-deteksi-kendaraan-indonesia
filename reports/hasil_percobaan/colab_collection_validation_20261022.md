# Colab collection — local verification, 22 October 2026

Status: **PASS LOCAL VERIFICATION**. **Colab GPU execution NOT TESTED**. This report does not establish completion/stability of 100-epoch training, online Drive mounting, or online W&B uploads.

## Delivered artifacts

- [Colab index and instructions](../../notebooks/yolo26-kd-indotraffic/colab/README.md): 17 standalone notebooks, numbered 00–16, plus a machine-readable manifest. The saved EDA-results duplicate and AMD/DirectML–Roboflow notebook are excluded.
- Existing Kaggle source notebooks were not edited for this conversion. Generation checks all files in their source directory before/after; all 17 notebook SHA-256 values match the conversion manifest.
- Manifest SHA-256: `0c54da9f7e2b9fabe919d21a6f6a5f7e3d5a053d26d79dc6e53629c2587f996b`.
- HEAD/RM/Pearson defaults: 100 epochs, full training enabled, smoke disabled. Existing experiment IDs, artifact references, mathematical/model/trainer implementations, hyperparameters and seed schedules are retained.

## Verification results

**42 Colab-specific checks passed**, plus **2 existing EDA checks passed** in the final command (**44 passed, 8.98 seconds**):

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_colab_notebooks.py tests/test_eda_poltrada_v3.py::test_yaml_no_test_and_pairing tests/test_eda_poltrada_v3.py::test_dictionary_names_lists_and_fallback --basetemp=tmp/colab_pytest_acceptance_20261022_74a91 -q --tb=short --show-capture=no
```

Coverage includes schema/syntax and deterministic generation of all 17 notebooks; unique names and valid index links; source hashes and research configuration/AST parity; ZIPs at archive root or inside a wrapper folder; original image/label/YAML byte preservation; separate runtime YAML and ordered TXT split lists; native loader and EDA compatibility; YAML ambiguity, duplicate keys, missing paths, corrupt CRC, traversal/symlink/duplicate ZIP entries; Colab Secret environment precedence and redacted errors; missing-GPU guidance; exact startup cells with mocked Drive mounting/dependency installation and real local file operations; teacher schema/class/provenance audit after relocation; and trio training dispatch without extra smoke runs. Image-membership counting matches the pinned Ultralytics loader formats.

The broader regression run covered HEAD/RM, PKD, teacher-s, serial multi-seed, scratch baseline, CSAKD, tracking, and EDA suites. Deduplicating the Colab checks, **231 existing checks passed and 3 were skipped** across that run and the two EDA reruns above. Initially, two existing EDA fixtures could not create the sandbox's default temporary directory; using a workspace-local `--basetemp` resolved them without changing their code. The skipped cases are one teacher-inapplicable baseline case and two opt-in CUDA checks requiring actual Poltrada/teacher assets. Remaining warnings were THOP dependency deprecations.

Ruff passed for the generator, embedded runtime helper, and new test module. `git diff --check` passed.

## Runtime behavior and limits

Drive holds the unchanged dataset ZIP and persistent outputs. The notebook copies/extracts data locally under `/content`; derived runtime paths are separate from the original teacher dataset declaration. ZIP, source YAML and runtime YAML hashes, class order and resolved split counts are recorded in Drive and training W&B config. The user-provided ZIP has not been independently compared byte-for-byte against the previous Kaggle dataset.

Secrets are read from the environment or Colab Secrets; no key is embedded. Drive authorization and a GPU runtime remain user steps in Colab. Tests simulate the Colab adapters and do not authenticate Google/W&B or launch GPU training. Existing resume restrictions and `RESUME_NOT_VERIFIED` labels remain in force.
