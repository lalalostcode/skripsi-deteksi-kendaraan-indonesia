"""Validation declarations embedded verbatim in standalone LAMP V2 notebooks.

The source never imports repository modules at notebook runtime. Numerical exports
come from the actual Ultralytics validator; missing measurements remain null.
"""
# ruff: noqa: F821 -- Notebook configuration/helpers provide these runtime globals.


def validation_arguments(data_yaml, output_dir, label):
    """Identical FP32 protocol for baseline/pruned; plots also collect confusion counts."""
    return {"data": str(data_yaml), "split": "val", "imgsz": IMGSZ, "batch": BATCH,
            "workers": WORKERS, "device": str(TARGET_DEVICE), "conf": 0.001,
            "iou": 0.7, "max_det": 300, "quantize": None, "nms": None,
            "augment": False, "plots": True, "verbose": False, "save_json": False,
            "project": str(output_dir), "name": label, "exist_ok": False}


def _validation_number(value):
    """JSON scalar: unsupported/nonfinite auxiliary metrics are unavailable, not zero."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _validation_array(value):
    import numpy as np

    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=float)


def _validation_json_array(value):
    import numpy as np

    array = _validation_array(value)
    return np.where(np.isfinite(array), array, None).tolist()


def _validation_per_class(metric, class_names):
    """Map SDK result rows via ap_class_index, never by names or array position."""
    import numpy as np

    box = metric.box
    support = getattr(metric, "nt_per_class", None)
    if support is not None:
        support = _validation_array(support)
        if (support.shape != (len(class_names),) or not np.isfinite(support).all()
                or (support < 0).any() or not np.equal(support, np.floor(support)).all()):
            raise ValueError("Support kelas SDK tidak valid atau tidak cocok dengan jumlah kelas.")
    rows = [{"class_id": i, "name": name, "ap50": None, "ap50_95": None,
             "precision": None, "recall": None, "f1": None,
             "support": int(support[i]) if support is not None else None}
            for i, name in enumerate(class_names)]
    indices = list(getattr(box, "ap_class_index", []))
    validated_indices = []
    for value in indices:
        scalar = _validation_number(value)
        if scalar is None or scalar != int(scalar) or not 0 <= int(scalar) < len(rows):
            raise ValueError("Indeks kelas SDK tidak valid.")
        validated_indices.append(int(scalar))
    if len(set(validated_indices)) != len(validated_indices):
        raise ValueError("Indeks kelas SDK duplikat.")
    for offset, class_id in enumerate(validated_indices):
        for key, attribute in (("ap50", "ap50"), ("ap50_95", "ap"),
                               ("precision", "p"), ("recall", "r"), ("f1", "f1")):
            values = getattr(box, attribute, None)
            if values is not None and len(values) > offset:
                rows[class_id][key] = _validation_number(values[offset])
    return rows, validated_indices


def _validation_curves(metric, class_names, class_ids):
    """Keep SDK axes and class mapping, replacing nonfinite samples with JSON null."""
    import numpy as np

    try:
        curve_values = metric.curves_results
    except (AttributeError, TypeError, ValueError) as error:
        return {"status": "UNAVAILABLE", "reason": str(error), "curves": []}
    names = getattr(metric, "curves", [])
    records = []
    for index, values in enumerate(curve_values):
        record = {"name": names[index] if index < len(names) else f"SDK curve {index}",
                  "class_ids": list(class_ids), "class_names": [class_names[i] for i in class_ids],
                  "status": "UNAVAILABLE", "x": [], "y": []}
        try:
            x_value, y_value, x_label, y_label = values
            x, y = _validation_array(x_value), _validation_array(y_value)
            record.update({"x_label": str(x_label), "y_label": str(y_label)})
            if y.ndim == 1 and len(class_ids) == 1:
                y = y[None, :]
            if (not class_ids or x.ndim != 1 or not x.size or y.ndim != 2
                    or y.shape != (len(class_ids), x.size)):
                record["reason"] = "Dimensi kurva tidak memiliki pemetaan kelas SDK yang valid."
            else:
                record.update({"x": _validation_json_array(x), "y": _validation_json_array(y),
                               "status": "MEASURED" if np.isfinite(x).all() and np.isfinite(y).all()
                               else "PARTIAL"})
                if record["status"] == "PARTIAL":
                    record["reason"] = "Sampel SDK NaN/Inf disimpan sebagai null."
        except (TypeError, ValueError, OverflowError) as error:
            record["reason"] = str(error)
        records.append(record)
    states = {item["status"] for item in records}
    status = "MEASURED" if states == {"MEASURED"} else (
        "PARTIAL" if states & {"MEASURED", "PARTIAL"} else "UNAVAILABLE")
    return {"status": status, "curves": records}


def _validation_confusion_matrix(validator, class_names, arguments):
    """Export actual counts and the distinct matching/NMS IoU thresholds."""
    import inspect

    import numpy as np

    matrix_object = getattr(validator, "confusion_matrix", None)
    result = {"status": "UNAVAILABLE", "labels": [*class_names, "background"],
              "orientation": "rows=predicted, columns=true", "matrix": None,
              "normalized_by_true": None, "confidence_threshold": None,
              "matching_iou_threshold": None, "validation_iou": arguments["iou"],
              "normalization": "Divide each true-class column by its sum; empty columns stay zero."}
    if matrix_object is None or getattr(matrix_object, "matrix", None) is None:
        result["reason"] = "Validator tidak menyediakan confusion matrix."
        return result
    matrix = _validation_array(matrix_object.matrix)
    expected = len(class_names) + 1
    if (matrix.shape != (expected, expected) or not np.isfinite(matrix).all() or (matrix < 0).any()):
        raise ValueError("Confusion matrix SDK tidak valid atau class order berbeda.")
    threshold = _validation_number(getattr(validator, "confusion_matrix_conf", None))
    matching_iou = None
    try:
        parameter = inspect.signature(matrix_object.process_batch).parameters.get("iou_thres")
        if parameter is not None and parameter.default is not inspect.Parameter.empty:
            matching_iou = _validation_number(parameter.default)
    except (AttributeError, TypeError, ValueError):
        pass
    totals = matrix.sum(axis=0, keepdims=True)
    normalized = np.divide(matrix, totals, out=np.zeros_like(matrix), where=totals != 0)
    result.update({"status": "MEASURED", "matrix": matrix.tolist(),
                   "normalized_by_true": normalized.tolist(),
                   "zero_support_columns": [int(i) for i in np.flatnonzero(totals[0] == 0)],
                   "confidence_threshold": threshold, "matching_iou_threshold": matching_iou,
                   "matching_iou_source": "Signature default of actual ConfusionMatrix.process_batch; validator calls without iou_thres.",
                   "confidence_source": "Actual DetectionValidator.confusion_matrix_conf"})
    return result


def _validation_metric_semantics(metric):
    """Record the SDK best-F1 reporting point without confusing it with CM confidence."""
    import numpy as np
    from ultralytics.utils.metrics import smooth

    result = {"precision_recall_f1": "SDK values at the global confidence maximizing smoothed mean F1 (IoU=0.50).",
              "reporting_confidence": None,
              "reporting_confidence_source": "SDK px[smooth(f1_curve.mean(0), 0.1).argmax()]",
              "map_iou_thresholds": [round(0.5 + 0.05 * i, 2) for i in range(10)],
              "absent_class": "Support 0 denotes no ground truth; class metrics without an SDK row remain null.",
              "confusion_matrix": "A separate operating point: see confidence_threshold and matching_iou_threshold."}
    box = metric.box
    try:
        px, f1 = _validation_array(box.px), _validation_array(box.f1_curve)
        if (px.ndim == 1 and f1.ndim == 2 and f1.shape[0] > 0 and f1.shape[1] == len(px)
                and np.isfinite(px).all() and np.isfinite(f1).all()):
            result["reporting_confidence"] = float(px[int(smooth(f1.mean(0), 0.1).argmax())])
    except (AttributeError, TypeError, ValueError):
        pass
    return result


def evaluate_checkpoint(path, dataset, output_dir, label):
    """Evaluate a fresh wrapper; preserve SDK metrics, curves and populated confusion counts."""
    wrapper = YOLO(str(path), task="detect")
    assert_class_match(wrapper.names, dataset["info"]["names"])
    class_names = list(dataset["class_names"])
    assert_class_match(wrapper.names, class_names)
    arguments = validation_arguments(dataset["yaml"], output_dir, label)
    captured = {}
    capture_validator = None
    if hasattr(wrapper, "add_callback"):
        # Model.val() keeps its validator local in Ultralytics 8.4.155.
        # Read the actual instance at callback time rather than assuming an attribute.
        def capture_validator(validator):
            captured["validator"] = validator

        wrapper.add_callback("on_val_end", capture_validator)
    try:
        metric = wrapper.val(**arguments)
    finally:
        # Avoid a callback -> capture -> validator cycle retaining GPU allocations
        # between baseline/ratio runs. Removing it also affects the shared callbacks.
        callbacks = getattr(wrapper, "callbacks", {}).get("on_val_end", [])
        if capture_validator in callbacks:
            callbacks.remove(capture_validator)
    box = metric.box
    per_class, class_ids = _validation_per_class(metric, class_names)
    result = {"map50_95": _validation_number(box.map), "map50": _validation_number(box.map50),
              "precision": _validation_number(box.mp), "recall": _validation_number(box.mr),
              "per_class": per_class, "split": "val", "precision_mode": "FP32", "nms": None}
    if any(result[key] is None for key in ("map50_95", "map50", "precision", "recall")):
        raise ValueError("Metrik evaluasi mengandung NaN/Inf atau tidak tersedia.")
    result["curves"] = _validation_curves(metric, class_names, class_ids)
    validator = captured.get("validator", getattr(wrapper, "validator", None))
    result["confusion_matrix"] = _validation_confusion_matrix(validator, class_names, arguments)
    result["metric_semantics"] = _validation_metric_semantics(metric)
    export_dir = Path(output_dir) / label
    export_dir.mkdir(parents=True, exist_ok=True)
    write_csv(export_dir / "per_class_metrics.csv", per_class)
    write_json(export_dir / "curves.json", result["curves"])
    write_json(export_dir / "confusion_matrix.json", result["confusion_matrix"])
    write_json(export_dir / "metric_semantics.json", result["metric_semantics"])
    result["exports"] = {name: str(export_dir / name) for name in (
        "per_class_metrics.csv", "curves.json", "confusion_matrix.json", "metric_semantics.json")}
    return result
