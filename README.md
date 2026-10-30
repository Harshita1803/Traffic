# DIP Smart Traffic Management: Adaptive Image Intelligence Workbench

**An Explainable Classical Digital Image Processing and Evaluation Framework for Adverse-Weather Traffic Surveillance**

[![Python 3.11+](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![Streamlit App](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Test Suite](https://img.shields.io/badge/pytest-115%20passed-brightgreen.svg)](https://pytest.org/)
[![License: CC BY-NC 3.0](https://img.shields.io/badge/dataset-DAWN%20(CC%20BY--NC%203.0)-yellow.svg)](https://data.mendeley.com/datasets/766ygrbt8y/3)

---

## 1. Project Overview & Problem Statement

Surveillance cameras in Intelligent Transportation Systems (ITS) frequently suffer from severe visual degradation caused by adverse meteorological conditions—including **dense fog, heavy rainfall, snowstorms, sandstorms, and low-light illumination**. These atmospheric phenomena cause:
- **Contrast flattening and veiling light** (Mie scattering from water droplets and aerosols).
- **High sensor noise and shadow loss** in low-light night-time conditions.
- **Obscuration of road markers, vehicle silhouettes, and license plates**.

While deep learning detectors are common in computer vision, running heavyweight neural networks directly on degraded, low-visibility video streams yields unreliable detections, high false negatives, and severe performance drops.

**DIP Smart Traffic Management** is a comprehensive, research-grade Digital Image Processing workbench designed to:
1. Objectively **measure** pixel-level degradation statistics (luminance distribution, contrast standard deviation, four-neighbour Laplacian variance, clipping, noise proxy, and dark-channel haze indicators).
2. Construct an **explainable, deterministic, rule-based adaptive enhancement plan** tailored specifically to the measured degradation.
3. Sequentially execute classical spatial-, luminance-, and frequency-domain image enhancement algorithms while preserving natural color chrominance.
4. Provide an interactive **Multi-Algorithm Comparative Benchmark** to systematically evaluate multiple algorithms against identical traffic scenes with precise microsecond timing.
5. Provide **ground-truth vehicle region evaluation** using authentic Pascal VOC XML annotations from the published DAWN dataset.
6. Support optional local **AI vehicle detection (YOLO)** with confidence tuning to examine downstream detection behavior without fabricating claims.

---

## 2. Mathematical Foundations & Algorithms

The system prioritizes **classical, mathematically grounded digital image processing techniques**. To prevent unnatural color distortions and chromatic shifts, spatial and contrast operations are performed on the luminance channel ($Y$) in the $YCbCr$ color space before recombining with chroma ($C_b, C_r$).

```
RGB Input  ──►  YCbCr Color Transform  ──►  Luminance Processing (Y)  ──►  RGB Synthesis & Validation
                                                     │
                             ┌───────────────────────┴───────────────────────┐
                             ▼                                               ▼
                  Spatial & Contrast Ops                          Frequency Domain
              (CLAHE, Gamma, DCP, Bilateral)                    (2D FFT Log-Spectrum, Lowpass)
```

### A. Color Representation & Conversion
Converting an $RGB$ image to $YCbCr$ isolates perceived luminance ($Y \in [0, 255]$) from blue-difference ($C_b \in [0, 255]$) and red-difference ($C_r \in [0, 255]$) chrominance:

$$Y = 0.299 R + 0.587 G + 0.114 B$$
$$C_b = 128 - 0.168736 R - 0.331264 G + 0.5 B$$
$$C_r = 128 + 0.5 R - 0.418688 G - 0.081312 B$$

Operating strictly on $Y$ preserves the original color balance and prevents the color distortion typical of naive independent-channel RGB stretching.

### B. Illumination & Dynamic Range Adjustment
1. **Robust Percentile Normalization**:
   Stretches a compressed luminance interval $[P_{low}, P_{high}]$ (default 1st to 99th percentiles) to $[0, 255]$:
   $$Y_{norm}(x, y) = \text{clip}\left( \frac{Y(x, y) - P_{low}}{P_{high} - P_{low}} \times 255, 0, 255 \right)$$
   Guarded against over-stretching when dynamic range is already broad.
2. **Power-Law Gamma Correction**:
   Corrects measured under-exposure or over-exposure:
   $$Y_{gamma}(x, y) = 255 \times \left( \frac{Y(x, y)}{255} \right)^\gamma$$
   Where $\gamma$ is calculated adaptively to map measured mean luminance $\bar{Y}$ toward a target operating level ($105$ for dark scenes, $165$ for over-exposed scenes):
   $$\gamma = \text{clip}\left( \frac{\ln(Y_{target} / 255)}{\ln(\bar{Y} / 255)}, 0.55, 1.60 \right)$$

### C. Contrast Restoration
1. **Global Histogram Equalization**:
   Redistributes global luminance values using the cumulative distribution function (CDF):
   $$s_k = T(r_k) = (L - 1) \sum_{j=0}^k p_r(r_j) = \frac{255}{N \times M} \sum_{j=0}^k n_j$$
2. **Contrast-Limited Adaptive Histogram Equalization (CLAHE)**:
   Overcomes global over-amplification by dividing $Y$ into localized contextual grid tiles (e.g. $64\times 64$). Tile histograms are computed, clipped at a predefined threshold $\beta$ (redistributing excess pixels uniformly across bins), equalized, and seamlessly blended across tile boundaries using **bilinear interpolation**:
   $$\beta = \max\left(1, \left\lfloor \text{clip\_limit} \times \frac{N_{tile}}{256} \right\rfloor\right)$$

### D. Dark-Channel Prior Dehazing
Implements the classical Koschmieder atmospheric scattering model:
$$I(x) = J(x) t(x) + A (1 - t(x))$$
Where $I(x)$ is observed hazy intensity, $J(x)$ is scene radiance, $A$ is atmospheric light, and $t(x)$ is medium transmission:
1. **Dark Channel Extraction**:
   $$J^{dark}(x) = \min_{c \in \{r, g, b\}} \left( \min_{y \in \Omega(x)} \frac{I^c(y)}{A^c} \right)$$
2. **Atmospheric Light Estimation ($A$)**:
   Selected from the top $0.1\%$ brightest pixels within the dark channel.
3. **Guarded Transmission Map**:
   $$t(x) = \max\left( t_0, 1 - \omega J^{dark}(x) \right)$$
   Where $\omega = 0.70$ controls haze retention and $t_0 = 0.55$ prevents severe darkening.
4. **Radiance Recovery**:
   $$J(x) = \frac{I(x) - A}{t(x)} + A$$

### E. Noise Reduction & Filtering
1. **Median Filter**: Non-linear filtering replacing each pixel with the median of its $3\times 3$ neighbourhood, highly effective for impulse and salt-and-pepper noise.
2. **Gaussian Filter**: Spatial domain low-pass convolution with a 2D Gaussian kernel of radius $r$.
3. **Bilateral Filter**: Non-linear, edge-preserving filter combining spatial geometric distance with photometric range intensity difference:
   $$I_{bilateral}(p) = \frac{1}{W_p} \sum_{q \in \Omega} I(q) \exp\left( -\frac{\|p - q\|^2}{2\sigma_s^2} \right) \exp\left( -\frac{\|I(p) - I(q)\|^2}{2\sigma_r^2} \right)$$
4. **2D Fourier Frequency-Domain Low-Pass**:
   Computes the 2D Fast Fourier Transform (FFT) $\mathcal{F}\{Y(x, y)\} = F(u, v)$, applies a Gaussian transfer function $H(u, v) = \exp(-2\pi^2 \sigma^2 (u^2 + v^2))$, and reconstructs via inverse 2D FFT.

### F. Detail Enhancement & Edge Sharpening
1. **Discrete Laplacian Variance Sharpness**:
   $$\nabla^2 Y(x, y) = Y(x+1, y) + Y(x-1, y) + Y(x, y+1) + Y(x, y-1) - 4 Y(x, y)$$
   $$\text{Sharpness} = \text{Var}(\nabla^2 Y)$$
2. **Luminance Unsharp Masking**:
   Subtracts a Gaussian blurred replica to isolate high-frequency edges and blends them back when exceeding threshold $\tau$:
   $$Y_{sharp} = Y + \alpha (Y - Y_{blur}) \quad \text{for } |Y - Y_{blur}| \ge \tau$$
3. **Bounded Sobel Detail Boost**:
   Computes horizontal and vertical Sobel gradients $G_x, G_y$ and adds bounded edge magnitude $\sqrt{G_x^2 + G_y^2}$ to the luminance channel.

### G. Image Segmentation & Thresholding
1. **Otsu's Optimal Automatic Thresholding**:
   Iterates through all possible intensity thresholds $t \in [0, 255]$ to maximize the inter-class variance $\sigma_b^2(t)$ between background ($\omega_0$) and foreground ($\omega_1$):
   $$\sigma_b^2(t) = \omega_0(t) \omega_1(t) \left[ \mu_0(t) - \mu_1(t) \right]^2 = \frac{\left[\mu_T \omega(t) - \mu(t)\right]^2}{\omega(t)(1 - \omega(t))}$$
   Outputs optimal threshold $T^*$ and a binary segmentation mask.
2. **Adaptive Local Thresholding**:
   Thresholds each pixel against the local mean of its neighbourhood block minus constant $C$.
3. **Mathematical Morphology**:
   Supports grayscale Erosion ($\epsilon$), Dilation ($\delta$), Opening ($\gamma = \delta \circ \epsilon$), Closing ($\phi = \epsilon \circ \delta$), **Morphological Gradient** ($\delta - \epsilon$ for vehicle boundary extraction), **Top-Hat** ($I - \gamma(I)$ for vehicle headlights and road markers), and **Black-Hat** ($\phi(I) - I$ for shadows and potholes).

---

## 3. System Architecture & Workspaces

The application is structured into modular layers that separate image-processing algorithms from presentation logic:

```
dip-smart-traffic-main/
├── app.py                      # Streamlit UI with navy-and-cyan responsive theme
├── requirements.txt            # Core dependencies (numpy, pillow, streamlit)
├── requirements-dev.txt        # Development dependencies (pytest)
├── requirements-ai.txt         # Optional AI extension (ultralytics)
├── src/
│   ├── analysis/
│   │   ├── quality.py          # Pixel metrics (brightness, contrast, sharpness, noise proxy)
│   │   └── degradation.py      # Heuristic degradation flags & evidence
│   ├── dip/
│   │   ├── adaptive.py         # Rule-based planner & pipeline executor
│   │   ├── benchmark.py        # Multi-algorithm comparative benchmark engine
│   │   ├── dehaze.py           # Koschmieder dark-channel prior model
│   │   ├── frequency.py        # 2D FFT log-spectrum & Fourier lowpass
│   │   └── spatial.py          # CLAHE, Otsu, morphology, bilateral, unsharp, resize
│   └── traffic_dip/
│       ├── batch.py            # Batch processing, decode, ZIP packaging
│       ├── detection.py        # Optional local YOLO vehicle detector
│       ├── evaluation.py       # Pascal VOC XML parser & fixed-IoU precision/recall
│       ├── presentation.py     # Pure presentation aggregators, table formatters, RGB hists
│       ├── reporting.py        # JSON & CSV report generators
│       └── visibility.py       # VOC vehicle-region metrics & box renderer
├── tests/                      # 115 unit & integration tests (100% passing)
└── data/
    ├── metadata/               # DAWN sample manifest (1,027 samples)
    ├── raw/dawn/images/        # Authentic Pascal VOC XML & image files
    └── samples/                # Bundled adverse-weather samples (fog, rain, snow, sand)
```

### The Three Interactive Workspaces

1. **Traffic Vision Lab (Control Room)**:
   - High-level command hero and project controls overview.
   - Real-time session KPI cards (Engine, Selection Policy, Session Status, Evaluation Status).
   - Interactive weather condition primer explaining degradation physics.
   - Bundled DAWN dataset sample gallery.
2. **Processing Studio**:
   - **Guided Single Image**:
     - Load an image either by direct file upload or by selecting from the **1,027 bundled DAWN adverse-weather samples**.
     - Choose **Adaptive Rule-Based** planning or **Fixed-Order** pipeline.
     - **Interactive Algorithm Tuning** sidebar with sliders for CLAHE clip limit & tile size, dehaze omega & floor, unsharp radius & percent, and percentile stretch.
     - Optional image pre-scaling (e.g. 1080p, 720p constraint) for memory safety.
     - Real-time step-by-step progress tracking with real computed pixels.
     - Serpentine stage timeline and draggable before/after divider comparison.
     - 3-tab image analysis: Luminance distribution, **RGB color-channel distributions**, and Spatial & Dynamic Range metrics.
     - **Ground-Truth Vehicle Region Viewer (Pascal VOC)** with authentic bounding boxes and inside-vehicle contrast/sharpness delta tables.
     - Optional CPU-based YOLO vehicle detector with confidence threshold slider.
   - **Algorithm Benchmark Matrix**:
     - Run multiple candidate algorithms (Normalization, CLAHE, Histogram Equalization, Dehazing, Bilateral Denoise, Median, Unsharp Mask, Sobel Edge, Otsu Binarization) on the same traffic input.
     - Side-by-side visual gallery with individual algorithm outputs.
     - Comparative metrics table and bar charts comparing execution time (ms) and contrast standard deviation ($\sigma$).
     - One-click CSV export of the complete benchmark matrix.
   - **Batch Processing**:
     - Process up to 10 images concurrently with individual stage logs.
     - Bundled DAWN batch selection or multi-file upload.
     - Downloadable ZIP containing original source bytes, individual stage PNGs, final PNGs, JSON reports, and CSV summaries.
3. **Results & Evaluation**:
   - Comprehensive summary of session visibility measurements.
   - Aggregated category statistics and methodology breakdown.
   - Exportable CSV reports (per-image, per-stage, vehicle-regions, and category summary).

---

## 4. Installation & Windows Run Instructions

### Prerequisites
- Windows 10 / 11 with PowerShell
- Python 3.11, 3.12, or 3.14

### Step-by-Step Setup

Open PowerShell and navigate to the project directory:

```powershell
cd C:\Users\harsh\Downloads\dip-smart-traffic-main\dip-smart-traffic-main
```

#### 1. (Optional) Create & Activate a Virtual Environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

#### 2. Install Required Dependencies
```powershell
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

#### 3. Run the Automated Test Suite
Verify that all 115 tests pass cleanly:
```powershell
python -m pytest -v
```

#### 4. Launch the Streamlit Application
```powershell
python -m streamlit run app.py
```

The application will start and automatically open in your default browser at:
`http://localhost:8501`

---

## 5. Optional AI Vehicle Detection Extension

Classical DIP operations run completely independently of any machine learning runtime. If you wish to enable the optional downstream YOLO vehicle detector:

```powershell
python -m pip install -r requirements-ai.txt
```

Place compatible pretrained YOLO weights at `models/yolo26n.pt` (or set the environment variable `$env:TRAFFIC_YOLO_MODEL = "C:\path\to\your\weights.pt"`).

> **Important College Demonstration Note**: An increase or decrease in vehicle detection counts between original and enhanced images does *not* prove detection improvement without paired clean-reference ground truth. The application displays count deltas transparently as observations, adhering strictly to scientific integrity.

---

## 6. Dataset Provenance & Attribution

The project includes sample imagery and annotations from the **DAWN (Vehicle Detection in Adverse Weather Nature)** dataset:
- **Authors**: Mourad A. Kenk and Mahmoud Hassaballah
- **Citation**: *DAWN: Vehicle Detection in Adverse Weather Nature Dataset*, Mendeley Data v3, DOI: [10.17632/766ygrbt8y.3](https://data.mendeley.com/datasets/766ygrbt8y/3); arXiv: [2008.05402](https://arxiv.org/abs/2008.05402).
- **License**: Creative Commons Attribution-NonCommercial 3.0 Unported (**CC BY-NC 3.0**).
- **Contents**: 1,027 adverse-weather images across four primary meteorological categories: Fog (300), Rain (200), Snow (204), and Sandstorm (323), paired with authentic Pascal VOC XML annotations.

---

## 7. Current Limitations & Future Work

- **No Aligned Clear-Weather References**: The DAWN dataset provides adverse-weather captures without paired clear-sky ground truth of the identical scene; hence PSNR and SSIM are omitted to prevent misleading claims.
- **Heuristic Indicators**: The dark-channel haze indicator and median-residual noise proxy are texture-sensitive mathematical proxies, not weather classification probabilities.
- **Future Work**: Integration of rain-streak physical removal models (e.g. directional derivative filtering) and paired clean/degraded traffic datasets (such as ACDC or synthetic weather simulators) for benchmark PSNR/SSIM validation.
