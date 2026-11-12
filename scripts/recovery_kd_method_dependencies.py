"""Resolve existing KD implementations at notebook generation time."""
import ast
import json
import runpy


def method_dependencies(root, method, *, source_matched=False):
    if method in {"finetune", "native"}:
        return []
    if method == "mgd":
        source = runpy.run_path(str(root / "scripts/mgd_method_source.py"))
        return [source["MGD_FEATURES_SOURCE"], source["MGD_LOSS_SOURCE"]]
    if method == "crosskd":
        path = root / ("notebooks/kaggle/kd/teacher_m/06_crosskd_yolo26s.ipynb" if source_matched else
                       "notebooks/kaggle/kd/teacher_s/11_crosskd_yolo26n_teacher_yolo26s.ipynb")
        cells = ["".join(cell["source"]) for cell in json.loads(path.read_text(encoding="utf-8"))["cells"]
                 if cell["cell_type"] == "code"]
        dependencies = [next(source for source in cells if marker in source)
                        for marker in ("def combined_loss(", "class CrossKDModel(")]
        if source_matched:
            return ["\n\n".join(ast.unparse(node) for node in ast.parse(source).body
                                if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef))
                                or isinstance(node, ast.ClassDef) and node.name == "CrossKDModel")
                    for source in dependencies]
        return dependencies
    raise ValueError(f"Metode recovery tidak dikenal: {method}")
