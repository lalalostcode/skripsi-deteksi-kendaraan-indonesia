"""Synthetic software checks; these are not empirical results for Poltrada V3."""

import ast
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd
import pytest
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = (
    ROOT / "notebooks/kaggle/eda/00_eda_poltrada_v3.ipynb"
)
QA_ROOT = ROOT / "outputs/eda_poltrada_v3_validation"


def read_notebook():
    return nbformat.read(NOTEBOOK_PATH, as_version=4)


def definitions_namespace():
    """Import libraries then execute definitions, without running dataset analysis."""
    namespace = {}
    for cell in read_notebook().cells:
        if cell.cell_type != "code":
            continue
        tree = ast.parse(cell.source)
        selected = [
            node
            for node in tree.body
            if isinstance(
                node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef)
            )
        ]
        exec(
            compile(ast.Module(body=selected, type_ignores=[]), "definitions", "exec"),
            namespace,
        )
    namespace.update(
        ISSUE_ROWS=[],
        TABLES={},
        FIGURE_RECORDS=[],
        FINDINGS=[],
        IMAGE_EXTENSIONS={".jpg", ".jpeg", ".png", ".webp"},
    )
    namespace["UniqueKeyLoader"].add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        namespace["construct_unique_mapping"],
    )
    return namespace


@pytest.fixture
def definitions():
    return definitions_namespace()


def snapshot_files(directory: Path) -> dict[str, str]:
    return {
        path.relative_to(directory).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def write_image(path: Path, offset: int) -> None:
    """Generate a visibly synthetic gradient for plotting/decode checks only."""
    height, width = 60 + offset, 100 + offset
    y, x = np.indices((height, width))
    pixels = np.stack([(x + offset) % 256, (y * 2) % 256, (x + y) % 256], axis=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels.astype(np.uint8)).save(path)


def make_dataset(
    root: Path, include_test: bool = True, empty_labels: bool = False
) -> Path:
    """Create an isolated fixture; class names do not claim the Poltrada ontology."""
    splits = ["train", "val", "test"] if include_test else ["train", "val"]
    for split_index, split in enumerate(splits):
        for folder in ("images", "labels"):
            (root / split / folder).mkdir(parents=True, exist_ok=True)
        for index in range(3):
            name = f"sample_{index}"
            write_image(
                root / split / "images" / f"{name}.png", split_index * 10 + index
            )
            text = (
                ""
                if empty_labels or index == 0
                else (
                    "0 0.5 0.5 0.2 0.4\n1 0.5 0.5 0.2 0.4\n"
                    if index == 1
                    else "0 0.3 0.3 0.1 0.1\n0 0.3 0.3 0.1 0.1\n0 0.6 0.6 0.1 0.1\n"
                )
            )
            (root / split / "labels" / f"{name}.txt").write_text(text, encoding="utf-8")
    if not empty_labels:
        write_image(root / "train/images/missing.png", 41)
        write_image(root / "train/images/invalid.png", 42)
        (root / "train/labels/invalid.txt").write_text(
            "0 0.5 0.5 0.2 0.2\n0 0.4 0.4\n0 nan 0.5 0.2 0.2\n"
            "9 0.5 0.5 0.1 0.1\n0 -0.1 0.5 0.1 0.1\n"
            "0 0.5 0.5 0 0.1\n0 0.99 0.5 0.3 0.2\n",
            encoding="utf-8",
        )
        (root / "train/images/corrupt.jpg").write_bytes(b"THIS IS NOT AN IMAGE")
        (root / "train/labels/corrupt.txt").write_text(
            "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
        )
        (root / "train/labels/orphan.txt").write_text(
            "1 0.5 0.5 0.1 0.1\n", encoding="utf-8"
        )
        (root / "val/images/shared.png").write_bytes(
            (root / "train/images/sample_1.png").read_bytes()
        )
        (root / "val/labels/shared.txt").write_text(
            "1 0.5 0.5 0.1 0.1\n", encoding="utf-8"
        )
    config = {split: f"{split}/images" for split in splits}
    config.update(
        nc=3, names=["Kelas Uji A", "Kelas Uji B", "Kelas Uji Tanpa Instance"]
    )
    yaml_path = root / "data.yaml"
    yaml_path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    return yaml_path


def run_all(yaml_path: Path, output: Path, mode: str, near: bool = False):
    """Run every original cell in order, injecting only the central configuration."""
    namespace = {"__name__": "__main__"}
    notebook = read_notebook()
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        source = cell.source
        if "configuration" in cell.metadata.get("tags", []):
            source += (
                f"\nDATA_YAML = Path({str(yaml_path)!r})"
                f"\nOUTPUT_DIR = Path({str(output)!r})"
                f"\nEDA_MODE = {mode!r}"
                f"\nRUN_NEAR_DUPLICATES = {near!r}"
                "\nDATASET_NAME = 'FIXTURE SINTETIK — UJI PERANGKAT LUNAK'"
                "\nBRIGHTNESS_SAMPLE_PER_SPLIT = 2"
                "\nNEAR_DUPLICATE_MAX_PAIRS = 2"
            )
        if "environment" in cell.metadata.get("tags", []):
            source += "\ndisplay = lambda *args, **kwargs: None\n"
        exec(compile(source, f"{NOTEBOOK_PATH.name}:cell_{index}", "exec"), namespace)
    return namespace


def test_notebook_schema_syntax_and_scope():
    notebook = read_notebook()
    nbformat.validate(notebook)
    sources = [cell.source for cell in notebook.cells if cell.cell_type == "code"]
    for index, source in enumerate(sources):
        compile(source, f"cell_{index}", "exec")
    assert all(
        cell.execution_count is None and not cell.outputs
        for cell in notebook.cells
        if cell.cell_type == "code"
    )
    markdown = "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "markdown"
    )
    for section in range(1, 17):
        assert f"## {section}. " in markdown
    for analysis_id in range(1, 17):
        assert f"EDA-{analysis_id:02d}" in markdown
    for source in sources:
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(
                    alias.name.split(".")[0] not in {"torch", "wandb", "ultralytics"}
                    for alias in node.names
                )
            if isinstance(node, ast.ImportFrom):
                assert node.module.split(".")[0] not in {
                    "torch",
                    "wandb",
                    "ultralytics",
                }
    assert len(sources) > 20


@pytest.mark.parametrize(
    ("text", "expected_issue"),
    [
        ("0 0.5 0.5 0.2", "jumlah_kolom_salah"),
        ("0 word 0.5 0.2 0.2", "nilai_bukan_numerik"),
        ("0 nan 0.5 0.2 0.2", "nilai_nan_inf"),
        ("0 0.5 inf 0.2 0.2", "nilai_nan_inf"),
        ("0.5 0.5 0.5 0.2 0.2", "class_id_bukan_integer"),
        ("9 0.5 0.5 0.2 0.2", "class_id_di_luar_rentang"),
        ("-1 0.5 0.5 0.2 0.2", "class_id_di_luar_rentang"),
        ("0 -0.2 0.5 0.1 0.1", "koordinat_normalisasi_tidak_valid"),
        ("0 0.5 0.5 0 0.1", "dimensi_nol_atau_negatif"),
        ("0 0.5 0.5 -0.1 0.1", "dimensi_nol_atau_negatif"),
        ("0 0.99 0.5 0.3 0.2", "bbox_melewati_batas"),
    ],
)
def test_invalid_parser_cases(definitions, text, expected_issue):
    records, issues = definitions["parse_yolo_text"](text, 3)
    assert len(records) == 1
    assert not records[0]["valid"]
    assert expected_issue in {issue["issue"] for issue in issues}
    assert records[0]["raw_text"] == text


def test_valid_empty_and_duplicate_parser(definitions):
    parser = definitions["parse_yolo_text"]
    assert parser("\n \n", 2) == ([], [])
    records, issues = parser("0 0.5 0.5 0.2 0.4\n0 0.5 0.5 0.2 0.4", 2)
    assert all(record["valid"] for record in records)
    assert not records[0]["duplicate"]
    assert records[1]["duplicate"]
    assert len(issues) == 1
    assert issues[0]["severity"] == "WARNING"


def test_coordinate_geometry_and_pairwise(definitions):
    assert definitions["xywh_to_pixels"](0.5, 0.5, 0.2, 0.4, 100, 50) == pytest.approx(
        (40, 15, 60, 35)
    )
    geometry = definitions["bbox_geometry"](0.2, 0.4, 100, 50)
    assert geometry["area_relative"] == pytest.approx(0.08)
    assert geometry["area_px"] == pytest.approx(400)
    assert geometry["aspect_ratio_pixel"] == pytest.approx(1)
    assert geometry["aspect_ratio_normalized"] == pytest.approx(0.5)
    frame = pd.DataFrame(
        {
            "x_center": [0.5, 0.5],
            "y_center": [0.5, 0.5],
            "width_norm": [0.2, 0.2],
            "height_norm": [0.4, 0.4],
        }
    )
    metrics, nearest = definitions["pairwise_geometry"](frame)
    assert metrics["pair_count"] == 1
    assert metrics["max_iou"] == pytest.approx(1)
    assert nearest.tolist() == [0, 0]
    frame.loc[1, ["x_center", "y_center"]] = [0.8, 0.8]
    metrics, nearest = definitions["pairwise_geometry"](frame)
    assert metrics["overlap_pairs"] == 0
    assert nearest[0] == pytest.approx(np.sqrt(0.18))


def test_descriptive_statistics_and_presence(definitions):
    result = definitions["describe_numeric"](
        pd.DataFrame({"value": [0, 1, 2, 3, np.nan]}), ["value"]
    ).iloc[0]
    assert result["count"] == 4
    assert result["median"] == pytest.approx(1.5)
    assert result["q1"] == pytest.approx(0.75)
    assert result["q3"] == pytest.approx(2.25)
    assert result["iqr"] == pytest.approx(1.5)
    assert result["std"] == pytest.approx(np.std([0, 1, 2, 3], ddof=1))
    assert result["p5"] == pytest.approx(0.15)
    boxes = pd.DataFrame({"image_id": ["a", "a", "a", "b"], "class_id": [0, 0, 1, 1]})
    presence = definitions["class_presence"](boxes, ["a", "b", "background"], [0, 1, 2])
    assert presence.to_numpy().tolist() == [[1, 1, 0], [0, 1, 0], [0, 0, 0]]
    assert (presence.to_numpy().T @ presence.to_numpy()).tolist() == [
        [1, 1, 0],
        [1, 2, 0],
        [0, 0, 0],
    ]


@pytest.mark.parametrize(
    "config",
    [
        {"nc": 2, "names": ["a"]},
        {"nc": 2, "names": {1: "a", 2: "b"}},
        {"nc": 2, "names": ["a", "a"]},
        {"nc": True, "names": ["a"]},
        {"nc": 2, "names": ["a", ""]},
    ],
)
def test_invalid_class_schema(definitions, config):
    with pytest.raises(ValueError):
        definitions["validate_class_schema"](config)


def test_yaml_no_test_and_pairing(definitions, tmp_path):
    yaml_path = make_dataset(tmp_path, include_test=False)
    data, root, names, splits, _, sources = definitions["load_dataset"](yaml_path)
    assert list(splits) == ["train", "val"]
    assert "test" not in data
    assert names == data["names"]
    assert root == tmp_path
    assert len(sources) == 2
    assert (
        definitions["image_to_label"](tmp_path / "train/images/nested/a.jpg")
        == tmp_path / "train/labels/nested/a.txt"
    )
    assert definitions["label_has_image_partner"](
        tmp_path / "train/labels/sample_1.txt"
    )
    assert not definitions["label_has_image_partner"](
        tmp_path / "train/labels/orphan.txt"
    )
    data["test"] = "missing/images"
    yaml_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="sumber hilang"):
        definitions["load_dataset"](yaml_path)


def test_dictionary_names_lists_and_fallback(definitions, tmp_path):
    for folder in ["train/images", "train/labels", "val/images", "val/labels"]:
        (tmp_path / folder).mkdir(parents=True)
    write_image(tmp_path / "train/images/a.png", 0)
    write_image(tmp_path / "val/images/b.png", 1)
    (tmp_path / "train.txt").write_text(
        "./train/images/a.png\n./train/images/a.png\n", encoding="utf-8"
    )
    data = {
        "nc": 2,
        "names": {"1": "b", "0": "a"},
        "train": ["train.txt"],
        "val": "../val/images",
    }
    yaml_path = tmp_path / "data.yaml"
    yaml_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    _, _, names, splits, _, sources = definitions["load_dataset"](yaml_path)
    assert names == ["a", "b"]
    assert len(splits["train"]) == 1
    assert "fallback" in sources.iloc[1]["resolusi"]
    issues = {row["issue"] for row in definitions["ISSUE_ROWS"]}
    assert {"fallback_path_yaml", "referensi_gambar_berulang"} <= issues
    yaml_path.write_text("nc: 2\nnc: 3\nnames: [a, b]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplikat"):
        definitions["load_dataset"](yaml_path)


@pytest.mark.parametrize(
    ("mode", "include_test", "empty_labels", "near"),
    [
        ("standard", False, False, False),
        ("full", True, False, True),
        ("standard", False, True, False),
    ],
)
def test_run_all_and_exports(mode, include_test, empty_labels, near):
    case = f"{mode}_{'test' if include_test else 'no_test'}_{'empty' if empty_labels else 'audit'}"
    dataset = QA_ROOT / case / "synthetic_input"
    output = QA_ROOT / case / "synthetic_output"
    yaml_path = make_dataset(dataset, include_test, empty_labels)
    before = snapshot_files(dataset)
    namespace = run_all(yaml_path, output, mode, near)
    assert snapshot_files(dataset) == before, (
        "Source dataset must remain byte-for-byte unchanged"
    )
    assert ("test" in namespace["SPLITS"]) == include_test
    assert plt.get_fignums() == []
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["dataset"].startswith("FIXTURE SINTETIK")
    assert manifest["runtime_seconds"] > 0
    assert len(namespace["FIGURE_RECORDS"]) >= 16
    figure_ids = {record["name"][:6] for record in namespace["FIGURE_RECORDS"]}
    for figure_id in [1, 2, 3, *range(5, 17)]:
        assert f"eda_{figure_id:02d}" in figure_ids
    assert (
        "class_imbalance" in namespace["TABLES"]
    )  # EDA-04 is a table by specification.
    for artifact in manifest["artifacts"]:
        path = output / artifact["path"]
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"]
    png_files = list((output / "figures").glob("*.png"))
    for path in png_files:
        with Image.open(path) as image:
            assert min(image.size) > 500
            assert image.info["dpi"][0] >= 299
            image.verify()
    json.loads(
        (output / "reports/eda_summary.json").read_text(encoding="utf-8"),
        parse_constant=lambda value: pytest.fail(value),
    )
    provenance = json.loads(
        (output / "reports/dataset_provenance.json").read_text(encoding="utf-8")
    )
    assert provenance["class_names"] == [
        "Kelas Uji A",
        "Kelas Uji B",
        "Kelas Uji Tanpa Instance",
    ]
    assert (
        provenance["yaml_sha256"] == hashlib.sha256(yaml_path.read_bytes()).hexdigest()
    )
    images = namespace["images_df"]
    assert images["brightness_sampled"].sum() <= len(images)
    if mode == "standard":
        assert not images["sha256"].notna().any()
        assert (
            images.loc[images["readable"], "integrity_level"]
            .eq("verifikasi struktur")
            .all()
        )
        assert images.groupby("split")["brightness_sampled"].sum().le(2).all()
    else:
        assert images["sha256"].notna().all()
        duplicate_groups = namespace["duplicates_df"]
        assert duplicate_groups.loc[
            duplicate_groups["method"].eq("sha256"), "cross_split"
        ].any()
        assert (
            duplicate_groups["label_byte_consistency"]
            .str.startswith("byte label berbeda")
            .any()
        )
        assert namespace["near_truncated"]
    if empty_labels:
        assert namespace["boxes_df"].empty
        assert images["n_objects"].eq(0).all()
        assert namespace["cooccurrence"].sum() == 0
    else:
        issues = {row["issue"] for row in namespace["ISSUE_ROWS"]}
        assert {
            "gambar_tanpa_label",
            "label_tanpa_gambar",
            "gambar_tidak_terbaca",
            "jumlah_kolom_salah",
            "nilai_nan_inf",
            "class_id_di_luar_rentang",
            "bbox_melewati_batas",
            "bbox_identik_terduplikasi",
        } <= issues
        missing = images.loc[images["image_path"].str.endswith("missing.png")].iloc[0]
        assert pd.isna(missing["n_objects"])
        assert not missing["annotation_complete"]
        assert int(namespace["class_total"]["instance"].sum()) == len(
            namespace["boxes_df"]
        )
        assert namespace["boxes_df"]["duplicate"].any()


def test_run_all_manifest_reproducibility():
    """Identical input and seed must produce identical tables/fingerprints/samples."""
    dataset = QA_ROOT / "reproducibility/synthetic_input"
    yaml_path = make_dataset(dataset, include_test=False)
    first = run_all(yaml_path, QA_ROOT / "reproducibility/run_a", "standard")
    second = run_all(yaml_path, QA_ROOT / "reproducibility/run_b", "standard")
    assert (
        first["provenance"]["inventory_fingerprint_sha256"]
        == second["provenance"]["inventory_fingerprint_sha256"]
    )
    for table in [
        "split_summary",
        "class_distribution",
        "brightness_coverage",
        "representative_samples",
        "spatial_bins",
    ]:
        pd.testing.assert_frame_equal(first["TABLES"][table], second["TABLES"][table])
