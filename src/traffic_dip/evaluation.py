"""Pascal VOC parsing, fixed-threshold evaluation, and multi-threshold mAP benchmarking."""
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from .detection import VEHICLE_CLASSES


def parse_voc_annotation(path):
    """Parse one DAWN Pascal VOC XML record into pixel-space xyxy boxes.

    The XML ``filename`` and dimensions are retained. Coordinates are returned
    as continuous ``(xmin, ymin, xmax, ymax)`` values exactly as stored in
    VOC; the paired DAWN YOLO annotations were audited and convert to the same
    coordinates. The source ``path`` field is deliberately ignored because it
    contains stale machine-local paths in the bundled dataset.
    """
    try:
        root = ET.parse(Path(path)).getroot()
    except (ET.ParseError, OSError) as exc:
        raise ValueError(f"Could not read Pascal VOC annotation {path}: {exc}") from exc
    filename = (root.findtext("filename") or "").strip()
    size = root.find("size")
    if not filename or size is None:
        raise ValueError(f"Pascal VOC annotation {path} is missing filename or size")
    try:
        width, height = int(size.findtext("width")), int(size.findtext("height"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Pascal VOC annotation {path} has invalid image dimensions") from exc
    if width <= 0 or height <= 0:
        raise ValueError(f"Pascal VOC annotation {path} has non-positive image dimensions")
    annotations = []
    for number, obj in enumerate(root.findall("object"), start=1):
        class_name = (obj.findtext("name") or "").strip().lower()
        box = obj.find("bndbox")
        if not class_name or box is None:
            raise ValueError(f"Pascal VOC annotation {path} object {number} is missing name or bndbox")
        try:
            coordinates = [float(box.findtext(key)) for key in ("xmin", "ymin", "xmax", "ymax")]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Pascal VOC annotation {path} object {number} has invalid coordinates") from exc
        x1, y1, x2, y2 = coordinates
        if not all(math.isfinite(value) for value in coordinates) or x2 <= x1 or y2 <= y1:
            raise ValueError(f"Pascal VOC annotation {path} object {number} has an invalid xyxy box")
        annotations.append({"class_name": class_name, "bbox_xyxy": coordinates,
                            "difficult": (obj.findtext("difficult") or "0").strip() == "1",
                            "truncated": (obj.findtext("truncated") or "0").strip() == "1"})
    return {"filename": filename, "width": width, "height": height, "annotations": annotations}


def intersection_over_union(box_a, box_b):
    """Compute IoU for continuous pixel-space xyxy boxes."""
    if len(box_a) != 4 or len(box_b) != 4:
        raise ValueError("IoU boxes must each contain four xyxy coordinates")
    ax1, ay1, ax2, ay2 = map(float, box_a)
    bx1, by1, bx2, by2 = map(float, box_b)
    if not all(math.isfinite(v) for v in (*box_a, *box_b)):
        raise ValueError("IoU coordinates must be finite")
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _scores(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"true_positive": tp, "false_positive": fp, "false_negative": fn,
            "precision": precision, "recall": recall, "f1": f1}


def evaluate_detection_records(records, iou_threshold=0.5, class_names=VEHICLE_CLASSES):
    """Aggregate class-aware precision/recall/F1 at one fixed IoU threshold.

    Each record has ``predictions`` (detection dictionaries from
    :func:`traffic_dip.detection.detect_vehicles`) and ``ground_truth`` (the
    annotations returned by :func:`parse_voc_annotation`). Predictions are
    greedily matched in descending confidence within each image and class.
    Difficult VOC objects are excluded from the denominator and absorb an
    overlapping prediction at threshold IoU. This is not an AP/mAP protocol.
    """
    if not 0 < float(iou_threshold) <= 1:
        raise ValueError("iou_threshold must be in (0, 1]")
    classes = tuple(str(name).lower() for name in class_names)
    if not classes or len(set(classes)) != len(classes):
        raise ValueError("class_names must be a non-empty sequence of unique names")
    totals = {name: [0, 0, 0] for name in classes}
    for record_index, record in enumerate(records, start=1):
        if not isinstance(record, dict) or "predictions" not in record or "ground_truth" not in record:
            raise ValueError(f"Evaluation record {record_index} requires predictions and ground_truth")
        gt_by_class = {name: [] for name in classes}
        difficult_by_class = {name: [] for name in classes}
        for gt in record["ground_truth"]:
            name = str(gt.get("class_name", "")).lower()
            if name not in gt_by_class:
                continue
            (difficult_by_class if gt.get("difficult", False) else gt_by_class)[name].append(gt["bbox_xyxy"])
        for name in classes:
            matched = set()
            predictions = sorted((p for p in record["predictions"]
                                  if str(p.get("class_name", "")).lower() == name),
                                 key=lambda p: float(p.get("confidence", 0)), reverse=True)
            for pred in predictions:
                box = pred.get("bbox_xyxy")
                if box is None or len(box) != 4:
                    raise ValueError(f"Evaluation prediction for {name} is missing a valid bbox_xyxy")
                candidates = [(intersection_over_union(box, gt), i)
                              for i, gt in enumerate(gt_by_class[name]) if i not in matched]
                best_iou, best_index = max(candidates, default=(0.0, None))
                if best_iou >= iou_threshold:
                    matched.add(best_index)
                    totals[name][0] += 1
                elif any(intersection_over_union(box, gt) >= iou_threshold for gt in difficult_by_class[name]):
                    continue
                else:
                    totals[name][1] += 1
            totals[name][2] += len(gt_by_class[name]) - len(matched)
    per_class = {name: _scores(*values) for name, values in totals.items()}
    aggregate = [sum(values[i] for values in totals.values()) for i in range(3)]
    return {"iou_threshold": float(iou_threshold), "class_names": list(classes),
            "per_class": per_class, "overall": _scores(*aggregate),
            "protocol": "Class-aware greedy one-to-one matching per image, confidence descending, fixed IoU threshold; not mAP."}


def compute_ap(recalls, precisions, method="11point"):
    """Compute Average Precision (AP) from sequences of recall and precision.

    Parameters
    ----------
    recalls : sequence of float
        Recall values in [0, 1].
    precisions : sequence of float
        Precision values in [0, 1].
    method : {"11point", "continuous"}
        - "11point": 11-point interpolated AP (Pascal VOC 2007 standard).
        - "continuous": Area under precision envelope (VOC 2010+ / COCO standard).

    Returns
    -------
    float
        Average Precision value in [0.0, 1.0].
    """
    if not len(recalls) or not len(precisions):
        return 0.0
    rec = np.array(recalls, dtype=np.float64)
    prec = np.array(precisions, dtype=np.float64)
    if method == "11point":
        ap = 0.0
        for t in np.linspace(0.0, 1.0, 11):
            matching = prec[rec >= t]
            p = float(np.max(matching)) if len(matching) else 0.0
            ap += p / 11.0
        return float(np.clip(ap, 0.0, 1.0))
    else:
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))
        for i in range(len(mpre) - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])
        idx = np.where(mrec[1:] != mrec[:-1])[0]
        ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
        return float(np.clip(ap, 0.0, 1.0))


def evaluate_detection_map(records, iou_thresholds=(0.5,), class_names=VEHICLE_CLASSES, ap_method="11point"):
    """Compute class-level AP, precision, recall, F1, false positives, missed vehicles,
    and mean Average Precision (mAP) across explicitly specified IoU thresholds.

    Parameters
    ----------
    records : list of dict
        Each record has 'predictions' (list of dict with 'class_name', 'confidence', 'bbox_xyxy')
        and 'ground_truth' (list of dict with 'class_name', 'bbox_xyxy', optional 'difficult').
    iou_thresholds : sequence of float
        IoU thresholds to evaluate (e.g. [0.5] or [0.5, 0.75]).
    class_names : sequence of str
        Target vehicle classes to include.
    ap_method : {"11point", "continuous"}
        Interpolation method for AP calculation.

    Returns
    -------
    dict
        Structured mAP report with per-threshold and per-class AP, precision, recall,
        false positives, missed vehicles, and overall mAP.
    """
    classes = tuple(str(c).lower() for c in class_names)
    thresholds = [float(t) for t in iou_thresholds]
    for t in thresholds:
        if not 0.0 < t <= 1.0:
            raise ValueError("All iou_thresholds must be in (0, 1]")

    map_by_thresh = {}
    detailed_by_thresh = {}

    for thresh in thresholds:
        class_metrics = {}
        aps = []
        for cname in classes:
            all_preds = []
            n_gt = 0
            gt_per_img = []
            diff_per_img = []

            for img_idx, rec in enumerate(records):
                gt_boxes = []
                diff_boxes = []
                for gt in rec.get("ground_truth", []):
                    if str(gt.get("class_name", "")).lower() == cname:
                        if gt.get("difficult", False):
                            diff_boxes.append(gt["bbox_xyxy"])
                        else:
                            gt_boxes.append(gt["bbox_xyxy"])
                n_gt += len(gt_boxes)
                gt_per_img.append(gt_boxes)
                diff_per_img.append(diff_boxes)

                for pred in rec.get("predictions", []):
                    if str(pred.get("class_name", "")).lower() == cname:
                        all_preds.append({
                            "img_idx": img_idx,
                            "confidence": float(pred.get("confidence", 0.0)),
                            "bbox_xyxy": pred["bbox_xyxy"],
                        })

            all_preds.sort(key=lambda p: p["confidence"], reverse=True)
            matched_per_img = [set() for _ in range(len(records))]

            tp = np.zeros(len(all_preds), dtype=np.float64)
            fp = np.zeros(len(all_preds), dtype=np.float64)

            for p_idx, pred in enumerate(all_preds):
                im_i = pred["img_idx"]
                p_box = pred["bbox_xyxy"]
                gt_list = gt_per_img[im_i]
                diff_list = diff_per_img[im_i]

                best_iou = 0.0
                best_gt_idx = -1
                for g_idx, g_box in enumerate(gt_list):
                    iou = intersection_over_union(p_box, g_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = g_idx

                if best_iou >= thresh:
                    if best_gt_idx not in matched_per_img[im_i]:
                        tp[p_idx] = 1.0
                        matched_per_img[im_i].add(best_gt_idx)
                    else:
                        fp[p_idx] = 1.0
                else:
                    is_diff = any(intersection_over_union(p_box, d_box) >= thresh for d_box in diff_list)
                    if not is_diff:
                        fp[p_idx] = 1.0

            cum_tp = np.cumsum(tp)
            cum_fp = np.cumsum(fp)
            recalls = cum_tp / n_gt if n_gt > 0 else np.zeros_like(cum_tp)
            precisions = cum_tp / (cum_tp + cum_fp) if len(cum_tp) else np.zeros_like(cum_tp)

            ap = compute_ap(recalls, precisions, method=ap_method)
            total_tp = int(cum_tp[-1]) if len(cum_tp) else 0
            total_fp = int(cum_fp[-1]) if len(cum_fp) else 0
            missed = max(0, n_gt - total_tp)

            final_prec = float(total_tp / (total_tp + total_fp)) if (total_tp + total_fp) > 0 else 0.0
            final_rec = float(total_tp / n_gt) if n_gt > 0 else 0.0
            f1 = float(2.0 * final_prec * final_rec / (final_prec + final_rec)) if (final_prec + final_rec) > 0 else 0.0

            class_metrics[cname] = {
                "ap": ap,
                "precision": final_prec,
                "recall": final_rec,
                "f1": f1,
                "true_positives": total_tp,
                "false_positives": total_fp,
                "missed_vehicles": missed,
                "ground_truth_count": n_gt,
                "prediction_count": len(all_preds),
            }
            if n_gt > 0:
                aps.append(ap)

        mean_ap = float(np.mean(aps)) if aps else 0.0
        map_by_thresh[str(thresh)] = mean_ap
        detailed_by_thresh[str(thresh)] = {
            "mAP": mean_ap,
            "classes": class_metrics,
            "total_false_positives": sum(m["false_positives"] for m in class_metrics.values()),
            "total_missed_vehicles": sum(m["missed_vehicles"] for m in class_metrics.values()),
        }

    overall_map = float(np.mean(list(map_by_thresh.values()))) if map_by_thresh else 0.0
    return {
        "overall_mAP": overall_map,
        "mAP_by_threshold": map_by_thresh,
        "threshold_evaluations": detailed_by_thresh,
        "evaluated_classes": list(classes),
        "total_records": len(records),
        "protocol": f"Multi-threshold class-aware mAP evaluation ({ap_method} AP interpolation); greedy matching.",
    }


def verify_bounding_box_scaling(original_dimensions, new_dimensions, bbox_xyxy):
    """Verify and scale a bounding box when image dimensions change.

    Returns the scaled (xmin, ymin, xmax, ymax) coordinates. Raises ValueError
    if aspect ratios differ by more than 0.1% or dimensions are non-positive.
    """
    ow, oh = original_dimensions
    nw, nh = new_dimensions
    if ow <= 0 or oh <= 0 or nw <= 0 or nh <= 0:
        raise ValueError("Image dimensions must be positive integers")
    orig_ratio = ow / oh
    new_ratio = nw / nh
    if not math.isclose(orig_ratio, new_ratio, rel_tol=1e-3, abs_tol=1e-6):
        raise ValueError(f"Aspect ratio mismatch: original {orig_ratio:.4f} vs new {new_ratio:.4f}")
    sx = nw / float(ow)
    sy = nh / float(oh)
    x1, y1, x2, y2 = bbox_xyxy
    scaled = [x1 * sx, y1 * sy, x2 * sx, y2 * sy]
    return [
        max(0.0, min(float(nw), scaled[0])),
        max(0.0, min(float(nh), scaled[1])),
        max(0.0, min(float(nw), scaled[2])),
        max(0.0, min(float(nh), scaled[3])),
    ]
