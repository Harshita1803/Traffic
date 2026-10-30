# Processing and Application Architecture

## 1. System Request Flow

```
┌────────────────────────────────────────────────────────────────────────┐
│                        Streamlit UI (app.py)                          │
│     [Traffic Vision Lab]  [Processing Studio]  [Results & Eval]        │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                       Data Decode & Validation                         │
│   • traffic_dip.batch.decode_image (EXIF normalize, mode provenance)   │
│   • dip.spatial.resize_preserving_aspect_ratio (memory safety)        │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
           ┌────────────────────────┴────────────────────────┐
           ▼                                                 ▼
┌───────────────────────────────┐     ┌───────────────────────────────────┐
│       Single & Batch DIP      │     │    Multi-Algorithm Benchmark      │
│  • analysis.quality           │     │  • dip.benchmark                  │
│  • analysis.degradation       │     │  • Parallel algorithm execution   │
│  • dip.adaptive.analyze_plan  │     │  • Microsecond execution timing   │
│  • dip.adaptive.build_pipeline│     │  • Comparative statistics table   │
└──────────────┬────────────────┘     └─────────────────┬─────────────────┘
               │                                        │
               ▼                                        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     Presentation & Export Layer                         │
│  • traffic_dip.presentation (cards, impact metrics, RGB histograms)     │
│  • traffic_dip.visibility (VOC vehicle regions, box rendering)          │
│  • traffic_dip.reporting (JSON reports, CSV metrics, ZIP archive)       │
│  • traffic_dip.detection (optional local YOLO CPU inference)            │
└─────────────────────────────────────────────────────────────────────────┘
```

1. **Input Ingestion & Validation**:
   - Streamlit accepts either a direct image upload (PNG, JPEG, BMP, TIFF, WebP) or one of the 1,027 bundled DAWN adverse-weather sample images.
   - `traffic_dip.batch.decode_image` validates file bytes, guards against decompression bombs, normalizes EXIF orientation, preserves original source mode, and optionally applies aspect-ratio-preserving constraint via `dip.spatial.resize_preserving_aspect_ratio`.
2. **Quality & Degradation Analysis**:
   - `analysis.quality.calculate_quality` computes pixel-derived metrics: mean brightness, contrast standard deviation, four-neighbour Laplacian variance, clipping fractions, channel statistics, saturation, dark-channel mean, local tile contrast, and median-residual noise proxy.
   - `analysis.degradation.assess_degradation` flags measurable degradation evidence with transparent heuristic thresholds.
3. **Adaptive Planning & Hyperparameter Tuning**:
   - `dip.adaptive.analyze_and_plan` creates a comprehensive, deterministic plan with an explicit entry for every candidate operation (selected/skipped/unavailable, rationale, parameters, known limitations).
   - Custom hyperparameter overrides from the interactive sidebar tuning panel (CLAHE clip limit/tile size, dehazing omega/transmission floor, unsharp masking radius/percent, percentile normalizations) are merged cleanly into the planned stage parameters.
4. **Pipeline Execution & Intermediate Retention**:
   - `dip.adaptive.build_pipeline` executes planned operations sequentially. Each operation consumes the exact computed output of the preceding stage.
   - The original input and each intermediate stage result are retained in memory with before/after measurements and changed pixel counts.
5. **Multi-Algorithm Benchmark Engine**:
   - `dip.benchmark.run_algorithm_benchmark` executes selected candidate algorithms against the same source image, recording exact microsecond processing durations and comparative metric shifts.
6. **Ground-Truth Vehicle Region Evaluation**:
   - `traffic_dip.visibility.evaluate_visibility` maps authentic Pascal VOC XML vehicle bounding boxes from the DAWN dataset, computing contrast and sharpness changes specifically within vehicle regions.
   - `traffic_dip.visibility.draw_ground_truth_boxes` renders verified vehicle bounding boxes with color-coded class tags onto both original and enhanced images.
7. **Downstream AI Detection**:
   - `traffic_dip.detection.detect_vehicles` wraps local YOLO inference on CPU with user-tunable confidence threshold, isolated from the classical DIP pipeline.
8. **Decoupled Presentation & Reporting**:
   - `traffic_dip.presentation` encapsulates all presentation data aggregation (conceptual stage cards, impact metrics, delta formatting, 32-bin RGB channel histograms, benchmark tables, and CSV bytes) with zero Streamlit dependencies.
   - `traffic_dip.reporting` generates schema-validated JSON reports and multi-stage CSV files.

---

## 2. Module Responsibilities

| Module | Purpose |
| :--- | :--- |
| `src/analysis/quality.py` | Pixel-derived image quality statistics (brightness, contrast, sharpness, noise proxy, dark-channel index). |
| `src/analysis/degradation.py` | Rule-based assessment of adverse-weather degradation flags. |
| `src/dip/spatial.py` | Spatial luminance transformations: robust stretch, gamma, equalize, CLAHE, median, Gaussian, bilateral, Otsu, adaptive thresholding, morphology, unsharp mask, aspect-ratio resize. |
| `src/dip/dehaze.py` | Koschmieder dark-channel prior atmospheric light and transmission recovery. |
| `src/dip/frequency.py` | Centered 2D FFT log-spectrum preview and 2D Gaussian Fourier low-pass filter. |
| `src/dip/adaptive.py` | Rule-based decision engine, stage normalizer, and sequential execution pipeline. |
| `src/dip/benchmark.py` | Multi-algorithm comparative benchmark engine with execution timing. |
| `src/traffic_dip/presentation.py`| UI presentation logic, stage cards, metric comparison rows, RGB histograms, benchmark table formatters. |
| `src/traffic_dip/batch.py` | Multi-image batch orchestration, decoding, decompression protection, ZIP archive creation. |
| `src/traffic_dip/visibility.py` | DAWN annotation matching, vehicle ROI crops, before/after region metrics, ground-truth box drawing. |
| `src/traffic_dip/evaluation.py` | Pascal VOC XML parsing, IoU computation, greedy class-aware precision/recall/F1 scoring. |
| `src/traffic_dip/detection.py` | Local YOLO model loader, CPU inference, bounding box drawing, detection comparison. |
| `src/traffic_dip/reporting.py` | JSON processing report and CSV batch export serialization. |
| `app.py` | Streamlit research interface with responsive navy-and-cyan design, interactive controls, and comparative views. |
