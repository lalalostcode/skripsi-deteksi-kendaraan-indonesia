"""Apply verified URI-only raw bindings without changing the fine-tuning implementation hash."""
import ast
import hashlib
import json
from pprint import pformat


def bind_restored_inputs(book, root, original_pins):
    receipt = root / "reports/pruning/lamp_yolo26s_ft_raw_bindings_20261108.json"
    if not receipt.exists():
        receipt = root / "reports/pruning/lamp_yolo26s_raw_restored_20261107.json"
    if not receipt.exists():
        return book
    proof = json.loads(receipt.read_text(encoding="utf-8"))
    if proof["status"] != "PASS":
        raise ValueError("Binding raw belum terverifikasi.")
    aliases, active, pins = {}, dict.fromkeys(original_pins), dict(original_pins)
    for row in proof["rows"]:
        old = row["historical_input"]
        pin = original_pins[old]
        expected = pin["baseline_ref"].rsplit("/", 1)[0]
        live = row["artifact_ref"].rsplit("/", 1)[-1]
        if (row["status"] != "CONFIRMED" or row["raw_sha256"] != pin["raw_sha256"]
                or row["manifest_sha256"] != pin["manifest_sha256"]
                or row["historical_artifact_ref"] != expected + "/" + old
                or row["artifact_ref"].rsplit("/", 1)[0] != expected
                or live.rsplit(":", 1)[0] != old.rsplit(":", 1)[0]):
            raise ValueError("Binding versi raw mengubah checkpoint yang dipatok.")
        aliases[row["historical_artifact_ref"]] = row["artifact_ref"]
        active[old] = live
        pins[live] = dict(pin)
    choices = [active[name] or name for name in original_pins]
    config = next(c for c in book["cells"] if "config" in c.get("metadata", {}).get("tags", []))
    source = "".join(config["source"])
    for old, live in active.items():
        if live:
            source = source.replace(f'"{old}"', f'"{live}"')
    config["source"] = source.splitlines(keepends=True)
    selection = next(c for c in book["cells"] if "selection" in c.get("metadata", {}).get("tags", []))
    source = "".join(selection["source"])
    begin, end = source.index("STUDENT_PINS ="), source.index("if len(STUDENT_INPUTS)")
    setup = "STUDENT_PINS = " + pformat(pins, sort_dicts=False, width=110)
    setup += "\nRAW_ARTIFACT_ALIASES = " + pformat(aliases, sort_dicts=False, width=110)
    setup += "\nACTIVE_STUDENT_INPUTS = " + repr(choices) + "\n"
    source = source[:begin] + setup + source[end:]
    source += '''
if RESUME_RUN_DIR:
    import json as _binding_json
    _path = Path(RESUME_RUN_DIR) / "resume_contract.json"
    if _path.is_file():
        _contract = _binding_json.loads(_path.read_text(encoding="utf-8"))["contract"]
        _old = _contract["raw_artifact_ref"].rsplit("/", 1)[-1]
        _selected = STUDENT_INPUTS[0]
        if (_old not in STUDENT_PINS or _old.rsplit(":", 1)[0] != _selected.rsplit(":", 1)[0]
                or _contract["raw_sha256"] != STUDENT_PINS[_selected]["raw_sha256"]):
            raise ValueError("Input resume berbeda dari raw pilihan.")
        STUDENT_INPUTS[:] = [_old]
        SEED = STUDENT_PINS[_old]["seed"]
'''
    selection["source"] = source.splitlines(keepends=True)
    runtime = next(c for c in book["cells"] if "runtime" in c.get("metadata", {}).get("tags", []))
    original = "".join(runtime["source"])
    nodes = ast.parse(original).body
    tracking = ast.unparse(next(n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == "ft_tracking"))
    tracking = tracking.replace('ft_identity()[\'raw_artifact_ref\'], type=\'model\'', "ft_resolved_raw_ref(), type='model'")
    tracking = tracking.replace("'study_settings': ft_study_settings()", "'resolved_raw_artifact_ref': ft_resolved_raw_ref(), 'artifact_binding_sha256': ARTIFACT_BINDING_SHA256, 'study_settings': ft_study_settings()")
    group = ast.unparse(next(n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == "ft_group_results"))
    group = group.replace("raw_ref.rsplit('/', 1)[-1]", "RAW_ARTIFACT_ALIASES.get(raw_ref, raw_ref).rsplit('/', 1)[-1]")
    adapter = (root / "scripts/finetune_artifact_binding_source.py").read_text(encoding="utf-8")
    binding_sha = hashlib.sha256((json.dumps(aliases, sort_keys=True) + adapter + tracking + group).encode()).hexdigest()
    # The original implementation hashes all training/evaluation code and pinned file content.
    # URI repair is an independently audited binding of the exact same files, including legacy resumes.
    extra = f"\nARTIFACT_BINDING_SHA256 = {binding_sha!r}\n" + adapter + "\n" + tracking + "\n" + group
    runtime["source"] = (original + extra).splitlines(keepends=True)
    book["metadata"]["research_workflow"]["artifact_binding_sha256"] = binding_sha
    for index, cell in enumerate(book["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"binding:{index}", "exec")
    return book
