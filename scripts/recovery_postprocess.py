"""Attach separately hashed I/O recovery without changing native training compatibility."""
import hashlib


def attach_postprocessing(book, root):
    source = (root / "scripts/recovery_postprocess_source.py").read_text(encoding="utf-8")
    signature = hashlib.sha256(source.encode()).hexdigest()
    runtime = next(cell for cell in book["cells"] if "runtime" in cell.get("metadata", {}).get("tags", []))
    extra = f"\nPOSTPROCESSING_SHA256 = {signature!r}\n" + source
    runtime["source"] = ("".join(runtime["source"]) + extra).splitlines(keepends=True)
    book["metadata"]["research_workflow"]["postprocessing_sha256"] = signature
    return book
