"""Research-grade Streamlit interface for the Smart Traffic DIP workbench."""
from base64 import b64encode
import csv
from html import escape
from hashlib import sha256
from pathlib import Path
import sys
import time

import numpy as np
import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
SRC_IMPORT_PATH = str(SRC.resolve())
# Put this checkout's source package first even when Streamlit or another
# extension has already added a different ``src`` directory later on sys.path.
if SRC_IMPORT_PATH in sys.path:
    sys.path.remove(SRC_IMPORT_PATH)
sys.path.insert(0, SRC_IMPORT_PATH)

import dip.adaptive
if Path(dip.adaptive.__file__).resolve() != (SRC / "dip" / "adaptive.py").resolve():
    raise ImportError(f"Smart Traffic DIP resolved dip.adaptive from an unexpected path: {dip.adaptive.__file__}")
from dip.adaptive import build_pipeline, normalize_pipeline_result
from dip.benchmark import AVAILABLE_BENCHMARK_OPERATIONS, run_algorithm_benchmark
from dip.frequency import spectrum_image
from dip.spatial import adaptive_threshold, otsu_threshold, resize_preserving_aspect_ratio
from traffic_dip.batch import (MAX_BATCH_IMAGES, MAX_IMAGE_PIXELS, batch_csv_bytes, batch_zip,
                               decode_image, png_bytes, process_batch)
from traffic_dip.detection import (DetectorInferenceError, DetectorUnavailableError,
                                   compare_detection_results, configured_model_path,
                                   detect_vehicles, is_detector_available)
from traffic_dip.presentation import (TECHNIQUE_GROUPS, benchmark_comparison_table,
                                      benchmark_csv_bytes, conceptual_stage_cards,
                                      photo_stages, rgb_histograms_32, stage_impact_data,
                                      stage_metric_rows, stage_output_caption, status_text,
                                      technique_group_status)
from traffic_dip.reporting import detection_comparison_csv_bytes, report_json_bytes
from traffic_dip.research_evaluation import (
    aggregate_category_statistics,
    category_summary_csv_bytes,
    create_run_manifest,
    evaluate_single_image,
    per_image_results_csv_bytes,
    research_report_json_bytes,
)
from traffic_dip.visibility import (build_dawn_annotation_index, draw_ground_truth_boxes,
                                    evaluate_visibility, find_dawn_annotation,
                                    visibility_category_csv_bytes, visibility_image_csv_bytes,
                                    visibility_region_csv_bytes, visibility_reports,
                                    visibility_stage_csv_bytes)

# Presentation helper aliases for direct backward compatibility
_photo_stages = photo_stages
_conceptual_stage_cards = conceptual_stage_cards
_stage_output_caption = stage_output_caption
_stage_metric_rows = stage_metric_rows
_stage_impact_data = stage_impact_data
_status_text = status_text
_technique_group_status = technique_group_status


SUPPORTED = ["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"]
CATEGORIES = ["unspecified", "fog", "rain", "snow", "sandstorm"]
PIPELINE_OPERATIONS = [
    "Robust luminance normalization", "Gamma correction", "Dark-channel-prior dehazing",
    "Global histogram equalization", "CLAHE on luminance", "Denoising",
    "Luminance unsharp mask", "Sobel edge enhancement",
]
OPERATION_PURPOSES = {
    "Robust luminance normalization": "Stretch compressed luminance percentiles while retaining color chroma.",
    "Gamma correction": "Correct measured under- or over-exposure using a power-law luminance transform.",
    "Dark-channel-prior dehazing": "Estimate atmospheric veil and transmission using a classical dark-channel prior.",
    "Global histogram equalization": "Redistribute global luminance values; can strongly change tone and amplify noise.",
    "CLAHE on luminance": "Improve local luminance contrast with clipped, interpolated tile histograms.",
    "Denoising": "Apply the selected median, Gaussian, bilateral, or frequency-domain filter.",
    "Luminance unsharp mask": "Increase moderate luminance edge contrast after checking the noise gate.",
    "Sobel edge enhancement": "Apply bounded Sobel-gradient emphasis while retaining a color photograph.",
}


def _inject_css():
    st.markdown("""
    <style>
    :root { color-scheme: dark; --canvas:#080f1c; --panel:#0f1a2b; --panel-2:#142238;
      --line:#22334a; --text:#edf5ff; --muted:#9aaec4; --accent:#48d8e8; --violet:#9ab7ff; }
    html, body, [data-testid="stAppViewContainer"] { background:radial-gradient(ellipse at 70% -20%, #142b46 0%, transparent 48%), var(--canvas); color:var(--text); }
    [data-testid="stHeader"] { background:rgba(16,18,22,.96); border-bottom:1px solid var(--line); }
    [data-testid="stSidebar"] { background:linear-gradient(180deg,#0d192a 0%,#0a1321 100%); border-right:1px solid var(--line); }
    /* Keep Streamlit's native top inset: its toolbar occupies this reserved area. */
    [data-testid="stMainBlockContainer"] { max-width:1480px; padding-bottom:3rem; }
    h1,h2,h3 { color:var(--text)!important; letter-spacing:-.025em; }
    h1 { font-size:2rem!important; font-weight:680!important; }
    h2 { font-size:1.35rem!important; font-weight:630!important; }
    h3 { font-size:1.05rem!important; font-weight:620!important; }
    p, label, .stMarkdown { color:var(--text); }
    small, [data-testid="stCaptionContainer"] { color:var(--muted)!important; }
    [data-testid="stMetric"] { background:linear-gradient(145deg,#13243a 0%,var(--panel) 74%); border:1px solid var(--line); border-radius:14px; padding:16px 18px; box-shadow:0 12px 28px rgba(0,0,0,.14); transition:transform .18s ease,border-color .18s ease; }
    [data-testid="stMetric"]:hover { transform:translateY(-2px); border-color:#31506b; }
    [data-testid="stMetricLabel"] { color:var(--muted)!important; }
    [data-testid="stMetricValue"] { color:var(--text)!important; font-size:1.25rem!important; }
    [data-testid="stVerticalBlockBorderWrapper"] > div { border-color:var(--line)!important; border-radius:14px!important; }
    [data-testid="stExpander"] { border:1px solid var(--line)!important; border-radius:12px!important; background:var(--panel); }
    [data-testid="stFileUploader"] section { background:#15191f; border:1px dashed #46515e; border-radius:14px; }
    [data-testid="stFileUploader"] section:hover { border-color:var(--accent); }
    .stButton button, .stDownloadButton button { background:#16253a; border:1px solid #2a405a; color:var(--text); border-radius:9px; min-height:2.5rem; }
    .stButton button:hover, .stDownloadButton button:hover { border-color:var(--accent); color:var(--accent); }
    .brand-row { display:flex; align-items:center; gap:13px; padding:4px 0 10px; }
    .brand-mark { display:grid; place-items:center; width:44px; height:44px; border-radius:13px; background:linear-gradient(145deg,#16394b,#12253d); color:var(--accent); font-weight:750; letter-spacing:-.08em; border:1px solid #276078; box-shadow:0 0 24px rgba(72,216,232,.1); }
    .brand-title { font-size:1.02rem; font-weight:680; color:var(--text); }
    .brand-sub { font-size:.77rem; color:var(--muted); margin-top:2px; }
    .eyebrow { text-transform:uppercase; letter-spacing:.12em; font-size:.68rem; color:var(--accent); font-weight:700; }
    .status-pill { display:inline-block; padding:5px 9px; border-radius:999px; background:#1b2928; border:1px solid #315451; color:#a0e8df; font-size:.76rem; }
    .command-hero { position:relative; overflow:hidden; padding:1.65rem 1.8rem; border:1px solid #27415c; border-left:3px solid var(--accent); border-radius:18px; background:linear-gradient(115deg,rgba(17,38,62,.98),rgba(13,24,40,.96) 58%,rgba(16,48,61,.72)); margin:.4rem 0 1.1rem; box-shadow:0 20px 48px rgba(0,0,0,.2); }
    .command-hero:after { content:""; position:absolute; right:-35px; top:-95px; width:260px; height:260px; border:1px solid rgba(72,216,232,.12); border-radius:50%; box-shadow:0 0 0 28px rgba(72,216,232,.025),0 0 0 58px rgba(72,216,232,.02); pointer-events:none; }
    .command-hero h1 { margin:.25rem 0 .55rem!important; font-size:2.05rem!important; }
    .command-hero p { color:#c1c8d1; max-width:900px; margin:0; line-height:1.55; }
    .section-kicker { color:var(--accent); text-transform:uppercase; letter-spacing:.11em; font-weight:700; font-size:.68rem; }
    .status-card { padding:14px 16px; border:1px solid var(--line); border-radius:13px; background:linear-gradient(145deg,#13243a 0%,var(--panel) 78%); min-height:82px; box-shadow:0 10px 24px rgba(0,0,0,.12); }
    .status-card .label { color:var(--muted); font-size:.72rem; text-transform:uppercase; letter-spacing:.08em; }
    .status-card .value { color:var(--text); font-size:.98rem; font-weight:650; margin-top:5px; }
    .traffic-step { min-height:138px; padding:14px; border:1px solid var(--line); border-radius:12px; background:var(--panel); }
    .traffic-step .number { color:var(--accent); font-size:.72rem; letter-spacing:.08em; font-weight:700; }
    .traffic-step .name { color:var(--text); font-weight:650; margin:.45rem 0 .25rem; }
    .traffic-step .detail { color:var(--muted); font-size:.8rem; line-height:1.45; }
    .implemented-tag { color:#91d8ba; }
    .planned-tag { color:#f1c47a; }
    .studio-hero { padding:1.5rem 1.7rem; border:1px solid var(--line); border-left:3px solid var(--accent); border-radius:14px; background:#111d2e; margin:.4rem 0 1rem; }
    .studio-hero h1 { margin:.25rem 0 .45rem!important; font-size:2.1rem!important; letter-spacing:.035em!important; }
    .studio-hero p { color:#c1c8d1; max-width:850px; margin:.3rem 0 0; line-height:1.55; }
    .technique-card { min-height:116px; padding:12px 14px; border:1px solid var(--line); border-radius:11px; background:var(--panel); margin-bottom:10px; }
    .technique-card strong { display:block; color:var(--text); margin-bottom:5px; }
    .technique-card span { display:block; color:var(--muted); font-size:.8rem; line-height:1.4; }
    .technique-status { display:inline-block!important; margin-top:9px; color:#9fe1c4!important; font-size:.68rem!important; text-transform:uppercase; letter-spacing:.08em; font-weight:700; }
    .technique-status.skipped { color:#aab3bd!important; }
    .technique-status.diagnostic { color:#c1b5ff!important; }
    .road-band { position:relative; display:flex; align-items:center; justify-content:space-between; gap:12px; overflow:hidden; padding:10px 14px; margin:.4rem 0 .6rem; min-height:44px; border:1px solid #434a53; border-radius:10px; background:#292e35; color:#eef0f2; }
    .road-band:before { content:""; position:absolute; left:10px; right:10px; top:50%; border-top:2px dashed #c9a958; opacity:.7; }
    .road-band span { position:relative; z-index:1; padding:2px 7px; border-radius:5px; background:#292e35; font-size:.77rem; font-weight:650; }
    .road-band .road-direction { color:#e8c878; white-space:nowrap; }
    .road-turn { margin:-.1rem .9rem .3rem; color:#aeb8c3; font-size:.75rem; }
    [class*="st-key-serpentine-row-"] [data-testid="stHorizontalBlock"] { align-items:stretch; }
    [class*="st-key-serpentine-row-"] [data-testid="stVerticalBlockBorderWrapper"] { background:#111d2e; }
    .workspace-heading { display:flex; align-items:baseline; justify-content:space-between; gap:1rem; }
    .condition-card { min-height:100px; padding:12px 14px; border:1px solid var(--line); border-radius:11px; background:#111d2e; margin-bottom:10px; }
    .condition-card strong { display:block; color:var(--accent); margin-bottom:5px; }
    .condition-card span { color:#b9c1ca; font-size:.82rem; line-height:1.45; }
    [class*="st-key-section-nav"] [data-testid="stRadio"] [role="radiogroup"] { gap:.35rem; }
    [class*="st-key-section-nav"] [data-testid="stRadio"] label { border:1px solid var(--line); border-radius:8px; padding:.35rem .7rem; background:#111d2e; }
    [class*="st-key-section-nav"] [data-testid="stRadio"] label:has(input:checked) { border-color:#28788a; background:#102c3b; }
    .step-row { padding:8px 10px; margin:4px 0; border-radius:8px; background:#191d23; border:1px solid var(--line); font-size:.83rem; }
    .step-row .muted { color:var(--muted); }
    .empty-card { border:1px solid var(--line); border-radius:16px; background:var(--panel); padding:30px; }
    [data-testid="stProgressBar"] > div > div { background:var(--accent); }
    div[data-baseweb="select"] > div, div[data-baseweb="input"] > div { background:#111d2e; border-color:var(--line); }
    [data-testid="stRadio"] label { color:var(--text); }
    [data-testid="stRadio"] [role="radiogroup"] { flex-wrap:wrap; gap:.25rem .75rem; }
    @media (max-width: 760px) { [data-testid="stMainBlockContainer"] { padding-left:1rem; padding-right:1rem; }
      h1 { font-size:1.55rem!important; } .brand-sub { max-width:65vw; }
      .brand-title { overflow-wrap:anywhere; }
      .command-hero { padding:1rem; } .command-hero h1 { font-size:1.55rem!important; }
      .traffic-step, .technique-card { min-height:0; }
      .road-band { align-items:flex-start; flex-direction:column; gap:4px; }
      .workspace-heading { align-items:flex-start; flex-direction:column; gap:.25rem; }
      [class*="st-key-serpentine-row-"] [data-testid="stHorizontalBlock"] { flex-direction:column!important; gap:.75rem!important; }
      [class*="st-key-serpentine-row-reverse-"] [data-testid="stHorizontalBlock"] { flex-direction:column-reverse!important; }
      [class*="st-key-serpentine-row-"] [data-testid="column"] { width:100%!important; flex:1 1 100%!important; min-width:100%!important; } }
    </style>
    """, unsafe_allow_html=True)


def _comparison_component(original, current, zoom="Fit to workspace", current_label="CURRENT OUTPUT"):
    original_data = b64encode(png_bytes(original.convert("RGB"))).decode("ascii")
    current_data = b64encode(png_bytes(current.convert("RGB"))).decode("ascii")
    height = 500
    html = f"""<!doctype html><html><head><meta charset="utf-8"><style>
    *{{box-sizing:border-box}} body{{margin:0;background:#111419;color:#e9edf1;font:13px system-ui,sans-serif}}
    .viewport{{height:{height}px;width:100%;overflow:auto;position:relative;background:#111419;border:1px solid #303640;border-radius:12px}}
    .canvas{{position:relative;margin:auto;overflow:hidden}}
    .canvas img{{position:absolute;inset:0;width:100%;height:100%;display:block}}
    #overlay{{position:absolute;inset:0;overflow:hidden;clip-path:inset(0 0 0 50%)}}
    .tag{{position:absolute;top:12px;padding:5px 8px;border-radius:6px;background:rgba(12,14,18,.84);z-index:2;color:#f3f5f7;font-size:12px}}
    #origTag{{left:12px}} #resultTag{{right:12px}}
    #divider{{position:absolute;top:0;bottom:0;left:50%;width:2px;background:#71ded5;z-index:3;pointer-events:auto;touch-action:none;cursor:ew-resize;box-shadow:0 0 0 1px rgba(0,0,0,.2)}}
    .sliderWrap{{position:sticky;left:0;bottom:0;width:100%;height:44px;padding:10px 18px;background:linear-gradient(transparent,rgba(10,12,15,.94));z-index:4}}
    input{{width:100%;accent-color:#48d8e8;cursor:ew-resize}} .help{{color:#aab3bd;font-size:11px;padding:0 4px 6px}}
    </style></head><body>
    <div class="help">Drag the divider or focus the slider and use the arrow keys · Original left / processed right</div>
    <div class="viewport" id="viewport" role="region" aria-label="Synchronized original and processed image comparison">
      <div class="canvas" id="canvas"><img alt="Original image" src="data:image/png;base64,{original_data}">
        <div id="overlay"><img alt="Processed image" src="data:image/png;base64,{current_data}"></div>
        <div class="tag" id="origTag">ORIGINAL</div><div class="tag" id="resultTag">{escape(current_label.upper())}</div><div id="divider"></div>
      </div>
      <div class="sliderWrap"><input id="range" type="range" min="0" max="100" value="50" aria-label="Move comparison divider" aria-valuetext="Divider at 50 percent from left"></div>
    </div>
    <script>
    const view=document.getElementById('viewport'), canvas=document.getElementById('canvas'), base=canvas.querySelector('img'),
      overlay=document.getElementById('overlay'), divider=document.getElementById('divider'), range=document.getElementById('range');
    function sizeCanvas(){{const w=base.naturalWidth||1,h=base.naturalHeight||1; const fit=Math.min((view.clientWidth-4)/w,({height}-48)/h);
      const actualZoom={'true' if zoom == 'Actual pixels' else 'false'}; const scale=actualZoom?1:fit; canvas.style.width=Math.max(1,Math.round(w*scale))+'px';canvas.style.height=Math.max(1,Math.round(h*scale))+'px';}}
    function setPos(){{const p=Number(range.value);overlay.style.clipPath='inset(0 0 0 '+p+'%)';divider.style.left=p+'%';range.setAttribute('aria-valuetext','Divider at '+p+' percent from left');}}
    let dragging=false; function pointerPos(e){{const rect=canvas.getBoundingClientRect();range.value=Math.max(0,Math.min(100,Math.round((e.clientX-rect.left)*100/rect.width)));setPos();}}
    divider.addEventListener('pointerdown',e=>{{dragging=true;divider.setPointerCapture(e.pointerId);pointerPos(e);}});
    divider.addEventListener('pointermove',e=>{{if(dragging)pointerPos(e);}});
    divider.addEventListener('pointerup',()=>{{dragging=false;}}); divider.addEventListener('pointercancel',()=>{{dragging=false;}});
    base.onload=sizeCanvas; window.addEventListener('resize',sizeCanvas); range.addEventListener('input',setPos); sizeCanvas();setPos();
    </script></body></html>"""
    components.html(html, height=height + 35, scrolling=False)


def _metrics_panel(metrics, label):
    st.markdown(f"**{label}**")
    cols = st.columns(3)
    cols[0].metric("Brightness", f"{metrics['mean_brightness']:.1f} / 255")
    cols[1].metric("Contrast · σ", f"{metrics['contrast_stddev']:.1f}")
    cols[2].metric("Sharpness · Laplacian var.", f"{metrics['sharpness_laplacian_variance']:.1f}")
    details = st.columns(4)
    details[0].caption(f"Local contrast {metrics['local_contrast_stddev']:.1f}")
    details[1].caption(f"Noise proxy {metrics.get('noise_estimate', 0):.2f}")
    details[2].caption(f"Shadow/highlight clipped {100*metrics['clipped_shadow_fraction']:.1f}% / {100*metrics['clipped_highlight_fraction']:.1f}%")
    details[3].caption(f"{metrics['width']}×{metrics['height']} · source {metrics['mode']}")
    with st.expander("Color, edge and haze-related measurements"):
        a, b, c = st.columns(3)
        a.write("**RGB means**")
        a.json(metrics["channel_means_rgb"])
        b.write("**RGB standard deviations**")
        b.json(metrics["channel_stddevs_rgb"])
        c.write("**Additional indicators**")
        c.write(f"Local contrast: {metrics['local_contrast_stddev']:.2f}")
        c.write(f"Mean edge gradient: {metrics['edge_gradient_mean']:.2f}")
        c.write(f"Edge fraction ≥ 4: {100*metrics['edge_fraction_gradient_ge_4']:.2f}%")
        c.write(f"Mean saturation: {metrics['mean_saturation']:.3f}")
        c.write(f"Haze-like indicator: {metrics['haze_like_indicator']:.3f} (heuristic, not a probability)")


def _show_histogram(original_metrics, current_metrics, before_image=None, after_image=None):
    st.markdown("**Image & channel distribution analysis**")
    tabs = st.tabs(["Luminance distribution", "RGB color channels", "Spatial & dynamic range"])
    with tabs[0]:
        st.line_chart({"Original": original_metrics.get("luminance_histogram_32", []),
                       "Current step": current_metrics.get("luminance_histogram_32", [])}, height=170)
        st.caption("32 bins · count per luminance range; a changed histogram does not alone prove visibility improved.")
    with tabs[1]:
        if before_image is not None and after_image is not None:
            before_rgb = rgb_histograms_32(before_image)
            after_rgb = rgb_histograms_32(after_image)
            col_r, col_g, col_b = st.columns(3)
            with col_r:
                col_r.caption("Red channel")
                col_r.line_chart({"Before": before_rgb["Red"], "After": after_rgb["Red"]}, height=130)
            with col_g:
                col_g.caption("Green channel")
                col_g.line_chart({"Before": before_rgb["Green"], "After": after_rgb["Green"]}, height=130)
            with col_b:
                col_b.caption("Blue channel")
                col_b.line_chart({"Before": before_rgb["Blue"], "After": after_rgb["Blue"]}, height=130)
            st.caption("RGB histograms reveal color cast corrections, channel clipping, and color balance.")
        else:
            st.caption("RGB image arrays not provided for channel histogram comparison.")
    with tabs[2]:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Robust Range", f"{current_metrics.get('robust_dynamic_range', 0.0):.1f}",
                  delta=f"{current_metrics.get('robust_dynamic_range', 0.0) - original_metrics.get('robust_dynamic_range', 0.0):+.1f}")
        c2.metric("Local Contrast σ", f"{current_metrics.get('local_contrast_stddev', 0.0):.1f}",
                  delta=f"{current_metrics.get('local_contrast_stddev', 0.0) - original_metrics.get('local_contrast_stddev', 0.0):+.1f}")
        c3.metric("Shadow Clip %", f"{100 * current_metrics.get('clipped_shadow_fraction', 0.0):.1f}%")
        c4.metric("Highlight Clip %", f"{100 * current_metrics.get('clipped_highlight_fraction', 0.0):.1f}%")


def _status_text(result):
    failed = sum(a["status"] == "failed" for a in result["plan"]["actions"])
    photo_count = sum(stage.get("output_kind") != "diagnostic" for stage in result["stages"])
    return f"{photo_count} filter(s) executed" + (f" · {failed} failed" if failed else "")


def _stage_output_caption(stage):
    """Format the canonical executed-stage identity and measured pixel delta."""
    return (f"Computed output of {stage['algorithm_id']} · "
            f"{stage['changed_pixel_fraction']:.1%} of pixels changed")


def _vehicle_pair(original, enhanced):
    before = detect_vehicles(original)
    after = detect_vehicles(enhanced)
    return {"original_detection": before, "enhanced_detection": after,
            "summary": compare_detection_results(before, after)}


def _detection_digest(images, salt=""):
    digest = sha256(salt.encode("utf-8"))
    for image in images:
        digest.update(str((image.mode, image.size)).encode("ascii"))
        digest.update(image.convert("RGB").tobytes())
    digest.update(str(configured_model_path()).encode("utf-8"))
    return digest.hexdigest()


def _render_single_vehicle_analysis(result, state_key, annotation=None):
    with st.expander("Traffic analysis & vehicle inspection · Pascal VOC & COCO detector", expanded=False):
        # 1. Authentic Ground-Truth Pascal VOC Vehicle Regions (DAWN dataset)
        if annotation and annotation.get("annotations"):
            st.markdown("#### Ground-Truth Vehicle Regions (Pascal VOC)")
            st.caption(f"Image: `{annotation.get('filename')}` · {annotation.get('width')}×{annotation.get('height')}px · Source: DAWN verified ground truth.")
            gt_orig = draw_ground_truth_boxes(result["original"], annotation)
            gt_final = draw_ground_truth_boxes(result["final"], annotation)
            col1, col2 = st.columns(2)
            col1.image(gt_orig, caption="Original with VOC Ground-Truth Boxes", width="stretch")
            col2.image(gt_final, caption="Enhanced with VOC Ground-Truth Boxes", width="stretch")
            
            gt_objs = [obj for obj in annotation.get("annotations", []) if obj.get("class_name") in {"car", "bus", "truck", "motorcycle", "bicycle"}]
            if gt_objs:
                st.markdown(f"**Verified vehicle objects:** {len(gt_objs)}")
                st.dataframe([{"Object #": i, "Class": obj["class_name"].title(), "Box [xmin, ymin, xmax, ymax]": [round(c, 1) for c in obj["bbox_xyxy"]], "Difficult": obj.get("difficult", False)}
                              for i, obj in enumerate(gt_objs, 1)], hide_index=True, width="stretch")
            st.divider()

        # 2. Optional local AI vehicle detector
        st.markdown("#### Optional AI Traffic Detector · YOLO on CPU")
        model_avail, model_status = is_detector_available()
        st.caption(f"Model status: `{model_status}`")
        if not model_avail:
            st.info("The optional YOLO detector is not installed or model weights were not found. "
                    "Classical DIP enhancement and Pascal VOC ground-truth evaluation operate independently without it. "
                    "To enable AI inference, install `requirements-ai.txt` and place weights at `models/yolo26n.pt`.")
            return

        enabled = st.checkbox("Run local vehicle detector on original and enhanced images",
                              key=state_key + "_ai_enabled")
        if not enabled:
            st.caption("Optional AI inference is off. Check the box above to run detection on CPU.")
            return

        conf_threshold = st.slider("Detection Confidence Threshold", min_value=0.10, max_value=0.90,
                                   value=0.25, step=0.05, key=state_key + "_ai_conf")
        digest = _detection_digest([result["original"], result["final"]], f"{state_key}_{conf_threshold}")
        cache_key = state_key + "_ai_result"
        cached = st.session_state.get(cache_key)
        if not cached or cached.get("digest") != digest:
            try:
                with st.spinner("Running local vehicle detector on both images…"):
                    before = detect_vehicles(result["original"], confidence=conf_threshold)
                    after = detect_vehicles(result["final"], confidence=conf_threshold)
                    pair = {"original_detection": before, "enhanced_detection": after,
                            "summary": compare_detection_results(before, after)}
                st.session_state[cache_key] = {"digest": digest, **pair}
                cached = st.session_state[cache_key]
            except DetectorUnavailableError as exc:
                st.warning(f"AI analysis is unavailable: {exc}")
                return
            except DetectorInferenceError as exc:
                st.error(f"Vehicle inference failed: {exc}")
                return
        original = cached["original_detection"]
        enhanced = cached["enhanced_detection"]
        summary = cached["summary"]
        left, right = st.columns(2)
        left.image(original["annotated_image"], caption=f"Original · {sum(original['counts'].values())} vehicle(s) · {original['inference_ms']:.1f} ms", width="stretch")
        right.image(enhanced["annotated_image"], caption=f"Enhanced · {sum(enhanced['counts'].values())} vehicle(s) · {enhanced['inference_ms']:.1f} ms", width="stretch")
        st.dataframe([{"Class": name.title(), "Original Count": summary["original_counts"][name],
                       "Enhanced Count": summary["enhanced_counts"][name], "Count Difference": summary["count_delta"][name]}
                      for name in summary["original_counts"]], hide_index=True, width="stretch")
        details = []
        for label, report in (("Original", original), ("Enhanced", enhanced)):
            details.extend({"Image": label, "Class": item["class_name"].title(),
                            "Confidence": f"{item['confidence']:.2f}", "Box [xyxy]": [round(c, 1) for c in item["bbox_xyxy"]]}
                           for item in report["detections"])
        if details:
            st.dataframe(details, hide_index=True, width="stretch")
        else:
            st.info(f"No vehicles detected at confidence ≥ {conf_threshold:.2f}. Try lowering the confidence threshold.")
        st.caption("A difference in detection counts is not a ground-truth accuracy measurement and does not show that enhancement improved detection.")


def _render_batch_vehicle_analysis(results, batch_signature):
    with st.expander("Optional batch AI vehicle counts", expanded=False):
        st.caption(f"Local model path: `{configured_model_path()}`. Inference uses CPU; weights are never downloaded automatically.")
        enabled = st.checkbox("Run vehicle analysis for each original/final pair", key="_batch_ai_enabled")
        if not enabled:
            return
        digest = sha256((batch_signature + str(configured_model_path())).encode("utf-8")).hexdigest()
        cached = st.session_state.get("_batch_ai_results")
        if not cached or cached.get("digest") != digest:
            rows, failures = [], []
            with st.spinner(f"Running local vehicle inference on {len(results)} original/final pairs…"):
                for item in results:
                    try:
                        pair = _vehicle_pair(item["original"], item["final"])
                        rows.append({"filename": item["filename"],
                                     "source_metadata": item.get("source_metadata", {}), **pair})
                    except DetectorUnavailableError as exc:
                        failures.append({"filename": item["filename"], "error": str(exc)})
                        break
                    except DetectorInferenceError as exc:
                        failures.append({"filename": item["filename"], "error": str(exc)})
            cached = {"digest": digest, "rows": rows, "failures": failures}
            st.session_state["_batch_ai_results"] = cached
        for failure in cached["failures"]:
            st.warning(f"AI analysis unavailable for {failure['filename']}: {failure['error']}")
        rows = cached["rows"]
        if rows:
            table = []
            for item in rows:
                summary = item["summary"]
                table.append({"Filename": item["filename"], "Original ms": item["original_detection"]["inference_ms"],
                              "Enhanced ms": item["enhanced_detection"]["inference_ms"],
                              **{f"Original {name}": summary["original_counts"][name] for name in summary["original_counts"]},
                              **{f"Enhanced {name}": summary["enhanced_counts"][name] for name in summary["enhanced_counts"]},
                              **{f"Δ {name}": summary["count_delta"][name] for name in summary["count_delta"]}})
            st.dataframe(table, hide_index=True, width="stretch")
            st.download_button("Download vehicle analysis · CSV", detection_comparison_csv_bytes(rows),
                               file_name="smart_traffic_vehicle_analysis.csv", mime="text/csv",
                               key="batch_vehicle_csv")
        st.caption("Detection counts are observations from a general COCO model, not ground-truth accuracy or incident detection.")


def _timeline(result):
    for action in result["plan"]["actions"]:
        status = action["status"].upper()
        accent = {"EXECUTED": "#81dfbd", "SKIPPED": "#9aa4b1", "FAILED": "#f29494", "UNAVAILABLE": "#a99cff"}.get(status, "#c9d0d9")
        st.markdown(f"<div class='step-row'><span style='color:{accent};font-weight:700'>{status}</span> &nbsp; {action['operation']}<br><span class='muted'>{action['reason']}</span></div>", unsafe_allow_html=True)


def _choose_stage(state_key, value):
    st.session_state[state_key + "_stage"] = value


def _photo_stages(result):
    """Return only executed photograph stages; diagnostics stay out of the main path."""
    return [stage for stage in result["stages"] if stage.get("output_kind") != "diagnostic"]


def _conceptual_stage_cards(original, stages):
    """Map N executed filters to N+1 cards with Stage 1 as the immutable source."""
    cards = [{"step": 1, "name": "Original", "image": original, "stage": None,
              "before_image": None, "status": "original"}]
    for number, stage in enumerate(stages, start=2):
        cards.append({"step": number, "name": stage["name"], "image": stage["image"],
                      "stage": stage, "before_image": stage["input_image"], "status": "completed"})
    return cards


TECHNIQUE_GROUPS = (
    ("Illumination correction", "Robust luminance normalization, Gamma correction",
     ("Robust luminance normalization", "Gamma correction"),
     "Adjusts luminance while retaining chroma where the selected method supports it."),
    ("Noise reduction", "Median, Gaussian, bilateral or FFT filtering",
     ("Median denoising", "Gaussian denoising", "Bilateral denoising", "FFT Gaussian low-pass", "Denoising"),
     "Selected from the noise proxy or explicit user settings; texture may also be smoothed."),
    ("Local contrast", "Global histogram equalization or CLAHE",
     ("Global histogram equalization", "CLAHE on luminance"),
     "Redistributes luminance; stronger contrast is not proof of better visibility."),
    ("Dehazing", "Dark-channel-prior atmospheric model",
     ("Dark-channel-prior dehazing",),
     "A classical heuristic selected only by measured evidence or explicit user choice."),
    ("Detail enhancement", "Unsharp masking or bounded Sobel emphasis",
     ("Luminance unsharp mask", "Sobel edge enhancement"),
     "Can emphasize edges, noise or compression artifacts; at most one is normally selected."),
    ("Structural detail analysis", "Sobel and morphology diagnostics",
     ("Sobel edge diagnostic", "Morphological diagnostic"),
     "Auxiliary outputs only; diagnostic images do not replace the traffic photograph."),
)


def _technique_group_status(result, operation_names):
    """Summarize only operations recorded by the real plan and stage result."""
    if result is None:
        return "Available", ""
    actions = [action for action in result.get("plan", {}).get("actions", [])
               if action.get("operation") in operation_names]
    stages = [stage for stage in result.get("stages", [])
              if stage.get("name") in operation_names]
    if any(stage.get("output_kind") == "diagnostic" for stage in stages):
        return "Diagnostic output", "diagnostic"
    if stages:
        return "Executed", ""
    if any(action.get("status") == "failed" for action in actions):
        return "Failed", "skipped"
    if actions and all(action.get("status") == "unavailable" for action in actions):
        return "Unavailable", "skipped"
    if actions:
        return "Skipped in this run", "skipped"
    return "Available", ""


def _render_technique_map(result=None):
    """Show the six supported technique families without implying execution."""
    st.markdown("#### Technique map · six supported method families")
    st.caption("Only stages with retained computed outputs are marked executed. Skipped and diagnostic methods remain distinct from the natural-color enhancement path.")
    for row_start in (0, 3):
        columns = st.columns(3)
        for column, (name, methods, operations, purpose) in zip(
                columns, TECHNIQUE_GROUPS[row_start:row_start + 3]):
            status, status_class = _technique_group_status(result, operations)
            safe_status = escape(status)
            with column:
                column.markdown(
                    f"<div class='technique-card'><strong>{escape(name)}</strong>"
                    f"<span>{escape(methods)}</span><span>{escape(purpose)}</span>"
                    f"<span class='technique-status {status_class}'>{safe_status}</span></div>",
                    unsafe_allow_html=True)


def _stage_metric_rows(stage):
    """Build compact comparisons from this operation's saved before/after metrics."""
    definitions = (("Mean luminance", "mean_brightness", "luminance"),
                   ("Global contrast · σ", "contrast_stddev", "luminance σ"),
                   ("Local contrast · σ", "local_contrast_stddev", "luminance σ"),
                   ("Sharpness · Laplacian variance", "sharpness_laplacian_variance", "variance"),
                   ("Noise proxy · median-residual MAD", "noise_estimate", "proxy"))
    rows = []
    for label, key, unit in definitions:
        before = stage.get("before", {}).get(key)
        after = stage.get("after", {}).get(key)
        if not isinstance(before, (int, float)) or not isinstance(after, (int, float)):
            continue
        delta = float(after) - float(before)
        relative = "n/a · zero baseline" if before == 0 else f"{delta / abs(float(before)):+.1%}"
        rows.append({"Measurement": label, "Before": f"{before:.3f} {unit}",
                     "After": f"{after:.3f} {unit}", "Absolute Δ": f"{delta:+.3f} {unit}",
                     "Relative change": relative})
    return rows


def _stage_impact_data(stage):
    """Return panel data from exactly one stage's saved input/output pair."""
    before, after = stage["input_image"], stage["image"]
    if before.size != after.size:
        changed_count, changed_fraction = None, None
    else:
        a = np.asarray(before.convert("RGB"), dtype=np.uint8)
        b = np.asarray(after.convert("RGB"), dtype=np.uint8)
        changed_count = int(np.any(a != b, axis=2).sum())
        changed_fraction = changed_count / (before.width * before.height)
    return {"name": stage["name"], "purpose": stage["explanation"],
            "reason": stage["reason"], "before_image": before, "after_image": after,
            "before_metrics": stage.get("before", {}), "after_metrics": stage.get("after", {}),
            "metric_rows": _stage_metric_rows(stage), "changed_pixel_count": changed_count,
            "changed_pixel_fraction": changed_fraction, "parameters": stage.get("parameters", {}),
            "processing_ms": stage.get("processing_ms"), "limitation": stage.get("limitation", "")}


def _render_stage_impact(stage):
    """Render the impact panel using only this stage's paired images and metrics."""
    impact = _stage_impact_data(stage)
    with st.expander("View filter impact", expanded=False):
        st.markdown(f"**{impact['name']} · Purpose:** {impact['purpose']}")
        st.markdown(f"**Why selected:** {impact['reason']}")
        before, after = st.columns(2)
        before.image(impact["before_image"], caption="Before · this filter's input", width="stretch")
        after.image(impact["after_image"], caption="After · this filter's output", width="stretch")
        if impact["metric_rows"]:
            st.dataframe(impact["metric_rows"], hide_index=True, width="stretch")
        else:
            st.caption("Quality measurements are unavailable for this stage.")
        if impact["changed_pixel_count"] is None:
            st.error("Changed-pixel count is unavailable because input and output dimensions differ.")
        else:
            pixels = impact["before_image"].width * impact["before_image"].height
            st.caption(f"Changed pixels: {impact['changed_pixel_count']:,}/{pixels:,} ({impact['changed_pixel_fraction']:.2%}). A pixel counts when any RGB channel differs.")
        duration = impact["processing_ms"]
        duration_text = "not recorded" if duration is None else f"{float(duration):.2f} ms"
        st.caption(f"Parameters: {impact['parameters']} · Duration: {duration_text} · Status: completed")
        st.caption(f"Limitations: {impact['limitation'] or 'None recorded.'}")
        st.caption("The noise measurement is a texture-sensitive proxy, not ground-truth noise reduction. Larger contrast or sharpness values alone do not establish better visibility.")


def _render_serpentine_timeline(original, stages, state_key, selected_index, options):
    """Display actual outputs in a directional, responsive serpentine route."""
    cards = _conceptual_stage_cards(original, stages)
    st.markdown(f"#### Executed stage route · {len(cards)} retained image stages")
    st.caption("The route follows the recorded execution order. Every thumbnail below is the original or an actual saved stage output.")
    for row_number, start in enumerate(range(0, len(cards), 2)):
        indices = list(range(start, min(start + 2, len(cards))))
        if row_number % 2:
            indices.reverse()
        direction = "reverse" if row_number % 2 else "forward"
        row_cards = [cards[index] for index in indices]
        route = f" {('←' if row_number % 2 else '→')} ".join(
            f"{card['step']:02d} · {escape(card['name'])}" for card in row_cards)
        st.markdown(
            f"<div class='road-band' aria-label='Processing route'>"
            f"<span>ASPHALT ROUTE</span><span>{route}</span>"
            f"<span class='road-direction'>{'← return lane' if row_number % 2 else 'forward lane →'}</span></div>",
            unsafe_allow_html=True)
        row_key = f"serpentine-row-{direction}-{row_number}-{sha256(state_key.encode()).hexdigest()[:8]}"
        with st.container(key=row_key):
            columns = st.columns(2)
            for column, card_index in zip(columns, indices):
                card = cards[card_index]
                with column.container(border=True):
                    selected = selected_index == card_index
                    st.caption(f"STEP {card['step']} · {card['status'].upper()}" + (" · SELECTED" if selected else ""))
                    st.image(card["image"], caption=card["name"], width="stretch")
                    st.caption(f"{card['image'].width} × {card['image'].height}px")
                    if card["stage"] is None:
                        st.caption("Unmodified upload · retained as the source for every filter comparison.")
                        st.button("Open original", key=f"{state_key}_timeline_{card_index}",
                                  on_click=_choose_stage, args=(state_key, options[card_index]),
                                  type="primary" if selected else "secondary")
                    else:
                        stage = card["stage"]
                        st.caption(f"{stage['algorithm_id']} · {stage['explanation']}")
                        st.button("Open this stage", key=f"{state_key}_timeline_{card_index}",
                                  on_click=_choose_stage, args=(state_key, options[card_index]),
                                  type="primary" if selected else "secondary")
        if start + 2 < len(cards):
            st.markdown("<div class='road-turn'>↳ turn at the end of this lane and continue the numbered route</div>",
                        unsafe_allow_html=True)


def _make_progress_callback(placeholder):
    """Render each computed stage as soon as its real output is retained."""
    progress = placeholder.progress(0.0, text="Analyzing source and preparing the operation plan…")
    preview = placeholder.empty()
    messages = []
    def update(event):
        kind = event["event"]
        total = event.get("total", 0)
        if kind == "plan_ready":
            messages.append(f"Plan ready · {total} selected operation(s)")
            progress.progress(0.0 if total else 1.0, text=f"Plan ready · {total} operation(s)")
            return
        if kind == "stage_started":
            number = event["operation_number"]
            fraction = (number - 1) / max(total, 1)
            progress.progress(fraction, text=f"Processing {number} of {total} · {event['operation']}")
            messages.append(f"Processing · {event['operation']}")
            current = event["current_image"]
            original = event["original"]
        elif kind == "stage_completed":
            number = event["operation_number"]
            progress.progress(number / max(total, 1), text=f"Completed {number} of {total} · {event['operation']}")
            stage = event["stage"]
            messages.append(f"✓ Step {stage['index']} · {stage['name']} · {stage['processing_ms']:.1f} ms")
            current, original = stage["image"], event["original"]
        elif kind in {"stage_skipped", "stage_failed"}:
            number = event["operation_number"]
            fraction = number / max(total, 1)
            progress.progress(fraction, text=f"{kind.replace('_', ' ').title()} · {event['operation']}")
            reason = event.get("reason", event.get("error", "No output stage was retained"))
            messages.append(f"{'↷' if kind == 'stage_skipped' else '×'} {event['operation']} · {reason}")
            current = event["current_image"]
            original = event.get("original", current)
        elif kind == "complete":
            progress.progress(1.0, text=f"Processing complete · {event['completed']} actual stage(s)")
            return
        else:
            return
        with preview.container(border=True):
            st.caption("Live processing preview · outputs below are actual computed pixels")
            left, right = st.columns(2)
            left.image(original, caption="Original · unchanged", width="stretch")
            right.image(current, caption="Latest completed output", width="stretch")
            st.markdown("\n".join(f"- {escape(message)}" for message in messages[-8:]))
    return update


def _make_batch_progress_callback(placeholder):
    """Display per-image and per-operation progress with the live actual output."""
    bar = placeholder.progress(0.0, text="Preparing batch…")
    preview = placeholder.empty()
    def update(event):
        image_index = event.get("image_index", 1)
        image_count = max(1, event.get("image_count", 1))
        stage_count = max(1, event.get("total", 1))
        operation_number = event.get("operation_number", 0)
        fraction = ((image_index - 1) + min(operation_number, stage_count) / stage_count) / image_count
        text = f"Image {image_index} of {image_count} · {event.get('filename', '')}"
        kind = event["event"]
        if kind == "stage_started":
            text += f" · processing {event['operation']}"
            current, original = event["current_image"], event["original"]
        elif kind == "stage_completed":
            text += f" · completed {event['operation']}"
            current, original = event["stage"]["image"], event["original"]
        elif kind in {"stage_skipped", "stage_failed"}:
            text += f" · {kind.replace('_', ' ')} {event['operation']}"
            current, original = event["current_image"], event.get("original", event["current_image"])
        elif kind == "complete":
            bar.progress(min(0.999, image_index / image_count),
                         text=f"Image {image_index} of {image_count} finalized · {event['completed']} stage(s)")
            return
        elif kind == "image_failed":
            bar.progress(min(0.999, image_index / image_count),
                         text=f"Image {image_index} of {image_count} failed · {event['filename']}")
            return
        elif kind == "batch_complete":
            bar.progress(1.0, text=f"Batch complete · {event['completed']} processed · {event['failed']} failed")
            return
        else:
            bar.progress((image_index - 1) / image_count, text=text + " · plan ready")
            return
        bar.progress(min(0.999, fraction), text=text)
        with preview.container(border=True):
            st.caption("Live batch preview · latest computed output")
            left, right = st.columns(2)
            left.image(original, caption="Original", width="stretch")
            right.image(current, caption="Latest completed output", width="stretch")
    return update


def _render_algorithm_benchmark(reset_nonce, custom_params=None):
    st.markdown("#### Multi-Algorithm Comparative Benchmark")
    st.caption("Evaluate multiple classical DIP algorithms side-by-side on the exact same traffic image. "
               "Compare execution time (ms), contrast enhancement, sharpness variance, and pixel deltas under identical test conditions.")
    
    bench_source = st.radio("Benchmark image source", ["Select a bundled DAWN sample", "Upload an image"],
                            horizontal=True, key=f"bench_source_{reset_nonce}")
    
    image = None
    source_name = ""
    
    if bench_source == "Select a bundled DAWN sample":
        manifest = _load_dawn_manifest()
        categories = sorted({row["category"] for row in manifest})
        if not manifest:
            st.error("DAWN manifest unavailable.")
            return
        c_col, f_col = st.columns([1, 2])
        chosen_cat = c_col.selectbox("Weather condition", categories, key=f"bench_dawn_cat_{reset_nonce}")
        rows = [r for r in manifest if r["category"] == chosen_cat]
        chosen_file = f_col.selectbox("Sample image", [r["filename"] for r in rows], key=f"bench_dawn_file_{reset_nonce}")
        row = next(r for r in rows if r["filename"] == chosen_file)
        try:
            path = _dawn_path(row)
            image = decode_image(path.read_bytes())
            source_name = chosen_file
        except Exception as exc:
            st.error(f"Could not load DAWN sample: {exc}")
            return
    else:
        upload = st.file_uploader("Upload a traffic image to benchmark", type=SUPPORTED, key=f"bench_upload_{reset_nonce}")
        if upload:
            try:
                image = decode_image(upload.getvalue())
                source_name = upload.name
            except Exception as exc:
                st.error(f"Could not load image: {exc}")
                return
        else:
            st.markdown("<div class='empty-card'><h3>Select or upload an image to begin benchmark</h3><p>Choose an image above to compare multiple image-processing techniques simultaneously.</p></div>", unsafe_allow_html=True)
            return

    if image is not None:
        st.markdown(f"**Loaded target:** `{source_name}` · {image.width}×{image.height}px")
        
        selected_ops = st.multiselect(
            "Select algorithms to benchmark",
            AVAILABLE_BENCHMARK_OPERATIONS,
            default=[
                "Robust Normalization",
                "Global Histogram Equalization",
                "CLAHE (Adaptive Equalization)",
                "Dark-Channel Dehazing",
                "Bilateral Denoising",
                "Median Denoising",
                "Luminance Unsharp Masking",
                "Sobel Edge Enhancement",
                "Otsu Automatic Binarization",
            ],
            key=f"bench_ops_{reset_nonce}"
        )
        
        if not selected_ops:
            st.warning("Please select at least one algorithm to benchmark.")
            return
            
        digest = sha256((source_name + str(image.size) + repr(selected_ops) + repr(custom_params) + str(reset_nonce)).encode()).hexdigest()[:12]
        cache_key = f"_benchmark_{digest}"
        cached = st.session_state.get(cache_key)
        
        run_btn = st.button("Run Comparative Benchmark", type="primary", key=f"run_bench_{reset_nonce}")
        
        if run_btn or cached:
            if not cached:
                with st.spinner(f"Running benchmark across {len(selected_ops)} algorithms on {source_name}…"):
                    bench_data = run_algorithm_benchmark(image, candidate_operations=selected_ops, custom_params=custom_params)
                    st.session_state[cache_key] = bench_data
            else:
                bench_data = cached
                
            results = bench_data["benchmark_results"]
            total_ms = bench_data["total_benchmark_ms"]
            
            non_baseline = [r for r in results if r["name"] != "Original (Baseline)"]
            fastest = min(non_baseline, key=lambda r: r["processing_ms"]) if non_baseline else results[0]
            max_contrast = max(non_baseline, key=lambda r: r["delta_contrast"]) if non_baseline else results[0]
            max_sharpness = max(non_baseline, key=lambda r: r["delta_sharpness"]) if non_baseline else results[0]
            
            kpi_cols = st.columns(4)
            kpi_cols[0].metric("Total Benchmark Time", f"{total_ms:.1f} ms", help="Sum of individual algorithm execution times")
            kpi_cols[1].metric("Fastest Algorithm", f"{fastest['name']}", f"{fastest['processing_ms']:.2f} ms")
            kpi_cols[2].metric("Peak Contrast Δ", f"{max_contrast['name']}", f"{max_contrast['delta_contrast']:+.1f} σ")
            kpi_cols[3].metric("Peak Sharpness Δ", f"{max_sharpness['name']}", f"{max_sharpness['delta_sharpness']:+.1f}")
            
            st.markdown("### Comparative Results Table")
            table_rows = benchmark_comparison_table(results)
            st.dataframe(table_rows, hide_index=True, width="stretch")
            
            st.download_button("Download Benchmark Matrix · CSV", benchmark_csv_bytes(results),
                               file_name=f"{Path(source_name).stem}_benchmark_matrix.csv", mime="text/csv",
                               key=f"bench_csv_download_{digest}")
            
            st.markdown("### Visual Comparison Gallery")
            st.caption("Actual computed outputs from each candidate algorithm applied to the same input.")
            cols_per_row = 3
            for i in range(0, len(results), cols_per_row):
                batch_items = results[i:i + cols_per_row]
                grid_cols = st.columns(len(batch_items))
                for col, item in zip(grid_cols, batch_items):
                    with col.container(border=True):
                        st.caption(f"**{item['name']}**")
                        st.image(item["image"], caption=f"{item['category']} · {item['processing_ms']:.1f} ms", width="stretch")
                        m = item.get("metrics", {})
                        st.caption(f"Contrast σ: {m.get('contrast_stddev', 0.0):.1f} ({item.get('delta_contrast', 0.0):+.1f}) · Sharpness: {m.get('sharpness_laplacian_variance', 0.0):.1f}")
            
            st.markdown("### Performance & Metric Visualizations")
            chart_col1, chart_col2 = st.columns(2)
            with chart_col1:
                st.caption("Execution Time (milliseconds) — lower is faster")
                timing_dict = {r["name"]: r["processing_ms"] for r in results}
                st.bar_chart(timing_dict, height=220)
            with chart_col2:
                st.caption("Global Contrast Standard Deviation (σ) — higher indicates wider tonal spread")
                contrast_dict = {r["name"]: r["metrics"]["contrast_stddev"] for r in results}
                st.bar_chart(contrast_dict, height=220)


def _load_dawn_manifest():
    path = ROOT / "data" / "metadata" / "dawn_sample_manifest.csv"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


@st.cache_data(show_spinner=False)
def _load_dawn_annotation_index():
    return build_dawn_annotation_index(ROOT / "data" / "raw" / "dawn" / "images")


def _dawn_path(row):
    path = (ROOT / "data" / "samples" / row["category"] / row["filename"]).resolve()
    sample_root = (ROOT / "data" / "samples").resolve()
    if sample_root not in path.parents or not path.is_file():
        raise ValueError(f"DAWN sample is missing or outside the sample directory: {row['filename']}")
    return path


def _get_pipeline(file_bytes, filename, metadata, denoise, dehaze, state_key,
                  edge_diagnostic=False, morphology_choice="off", selected_operations=None,
                  progress_callback=None, pipeline_mode="Adaptive", custom_parameters=None,
                  max_dim=None):
    options = repr((denoise, dehaze, edge_diagnostic, morphology_choice,
                    pipeline_mode, selected_operations, custom_parameters, max_dim)).encode()
    digest = sha256(file_bytes + repr(sorted(metadata.items())).encode() + options).hexdigest()
    if st.session_state.get(state_key + "_digest") != digest:
        image = decode_image(file_bytes, max_dim=max_dim)
        st.session_state[state_key + "_result"] = build_pipeline(image, denoise_choice=denoise,
                                                                 dehaze_choice=dehaze, source_metadata=metadata,
                                                                 edge_diagnostic=edge_diagnostic,
                                                                 morphology_choice=morphology_choice,
                                                                 selected_operations=selected_operations,
                                                                 progress_callback=progress_callback,
                                                                 custom_parameters=custom_parameters)
        st.session_state[state_key + "_digest"] = digest
        st.session_state[state_key + "_filename"] = filename
        stage_key = state_key + "_stage"
        stages = st.session_state[state_key + "_result"]["stages"]
        st.session_state[stage_key] = (f"01 · {stages[0]['name']}" if stages else "00 · Original")
    # Streamlit may retain a pre-schema result in an already-open session after
    # code reload. Normalize at the read boundary before any UI consumer sees it.
    result = normalize_pipeline_result(st.session_state[state_key + "_result"])
    st.session_state[state_key + "_result"] = result
    return result


def _render_result(result, filename, state_key, download_prefix="single"):
    st.markdown("#### Processing studio · sequential image outputs")
    actions = result["plan"]["actions"]
    executed = _photo_stages(result)
    diagnostics = [stage for stage in result["stages"] if stage.get("output_kind") == "diagnostic"]
    st.caption(f"{_status_text(result)} · {result['total_processing_ms']:.0f} ms total · original retained for comparison")
    options = ["00 · Original"] + [f"{i:02d} · {stage['name']}" for i, stage in enumerate(executed, start=1)]
    stage_key = state_key + "_stage"
    current_selection = st.session_state.get(stage_key, options[0])
    current_selection = current_selection if current_selection in options else options[0]
    selected_index = options.index(current_selection)
    _render_technique_map(result)
    _render_serpentine_timeline(result["original"], executed, state_key, selected_index, options)
    previous_col, next_col, picker_col = st.columns([.8, .8, 3])
    previous_col.button("← Previous Stage", disabled=selected_index == 0, key=state_key + "_previous",
                         on_click=_choose_stage, args=(state_key, options[max(0, selected_index - 1)]))
    next_col.button("Next Stage →", disabled=selected_index >= len(options) - 1, key=state_key + "_next",
                    on_click=_choose_stage, args=(state_key, options[min(len(options) - 1, selected_index + 1)]))
    choice = picker_col.selectbox("Jump to a completed stage", options, key=stage_key)
    index = options.index(choice)
    st.progress(index / max(len(executed), 1), text=("Source image · plan ready" if index == 0 else
                                                     f"Step {index} of {len(executed)} · {executed[index-1]['name']}"))
    with st.container(border=True):
        selected_title = "Stage 1 · Original" if index == 0 else f"Stage {index + 1} · {executed[index - 1]['name']}"
        st.markdown(f"### {selected_title}")
        zoom = st.radio("View", ["Fit to workspace", "Actual pixels"], horizontal=True,
                        key=state_key + "_zoom", label_visibility="collapsed")
        if index == 0:
            current_image = result["original"]
            stage = None
        else:
            stage = executed[index - 1]
            current_image = stage["image"]
        if stage:
            st.markdown(f"**{stage['name']}** · {stage['explanation']}")
            st.caption(f"Parameters: {stage['parameters']} · {stage['processing_ms']:.2f} ms · input {stage['input_image'].width}×{stage['input_image'].height}px")
            compare_mode = st.radio("Selected filter comparison", ["Side by side", "Draggable divider"],
                                    horizontal=True, key=state_key + "_compare", index=0)
            if compare_mode == "Draggable divider":
                _comparison_component(stage["input_image"], stage["image"], zoom, "After filter")
            else:
                before_after = st.columns(2)
                before_after[0].image(stage["input_image"], caption="Before · previous stage", width="stretch")
                before_after[1].image(stage["image"], caption=f"After · {stage['name']}", width="stretch")
            _render_stage_impact(stage)
            _show_histogram(stage["before"], stage["after"], before_image=stage["input_image"], after_image=stage["image"])
        else:
            st.image(current_image, caption="Original · no operation has been applied", width="stretch")
            _metrics_panel(result["input_metrics"], "Original measurements")
            _show_histogram(result["input_metrics"], result["input_metrics"], before_image=result["original"], after_image=result["original"])
    with st.expander("Analysis, evidence and skipped operations", expanded=False):
        assessment = result["analysis"]
        st.write("**Measured degradation indicators**")
        st.write(" · ".join(assessment["flags"]) if assessment["flags"] else "No configured indicator crossed its heuristic threshold.")
        st.info(assessment["evidence"]["interpretation"])
        st.write(f"Haze-like indicator: {assessment['metrics']['haze_like_indicator']:.3f} · dark-channel mean: {assessment['metrics']['dark_channel_mean']:.3f} · local-contrast evidence: {assessment['metrics']['haze_local_contrast_evidence']:.3f} · mean saturation: {assessment['metrics']['mean_saturation']:.3f}")
        st.write(f"Noise proxy: {assessment['noise_estimate']:.2f}. {assessment['evidence']['noise_interpretation']}")
        st.write("**Decision log**")
        _timeline(result)
    st.download_button("Download processing report · JSON", report_json_bytes(result, filename, result["plan"]["source_metadata"]),
                       file_name=Path(filename).stem + "_processing_report.json", mime="application/json",
                       key=download_prefix + "_report_" + state_key)
    visibility_digest = sha256(
        result["original"].tobytes()
        + repr([(stage["algorithm_id"], stage["parameters"], stage["changed_pixel_count"])
                for stage in result["stages"]]).encode()).hexdigest()[:12]
    visibility_key = state_key + "_visibility_" + visibility_digest
    visibility_report = st.session_state.get(visibility_key)
    if visibility_report is None:
        metadata = result["plan"].get("source_metadata", {})
        annotation = None
        if str(metadata.get("source_dataset", "")).lower() == "dawn":
            annotation = find_dawn_annotation(_load_dawn_annotation_index(), filename, metadata.get("category", ""))
        visibility_report = evaluate_visibility(result, filename, metadata, annotation)
        st.session_state[visibility_key] = visibility_report
    with st.expander("Visibility evaluation · measurements and indicators", expanded=False):
        st.caption(visibility_report["per_image"]["conclusion_note"])
        st.write("Final minus original · absolute metric changes")
        st.dataframe([{"Measurement": key, "Absolute change": value["absolute"],
                       "Relative change": value["relative"]}
                      for key, value in visibility_report["per_image"]["original_to_final_metric_differences"].items()],
                     hide_index=True, width="stretch")
        st.caption("Noise is a texture-sensitive proxy. Edge, local-detail and clipping changes are indicators, not a universal visibility score. Plate ground truth and aligned clear references are unavailable.")
        st.download_button("Download visibility summary · CSV",
                           visibility_image_csv_bytes([visibility_report]),
                           file_name=Path(filename).stem + "_visibility.csv", mime="text/csv",
                           key=visibility_key + "_image_csv")
        st.download_button("Download stage measurements · CSV",
                           visibility_stage_csv_bytes([visibility_report]),
                           file_name=Path(filename).stem + "_visibility_stages.csv", mime="text/csv",
                           key=visibility_key + "_stage_csv")
    st.markdown("#### Downloads")
    st.download_button("Download final image · PNG", png_bytes(result["final"]),
                       file_name=Path(filename).stem + "_enhanced.png", mime="image/png",
                       key=download_prefix + "_final_" + state_key)
    if index == len(executed):
        with st.expander("Original versus final output", expanded=False):
            f1, f2 = st.columns(2)
            f1.image(result["original"], caption="Original", width="stretch")
            f2.image(result["final"], caption="Final output", width="stretch")
        st.caption("No PSNR or SSIM is reported because this input has no aligned clear-weather reference.")
    if diagnostics:
        with st.expander("Advanced diagnostics · FFT, Sobel and morphology", expanded=False):
            for diagnostic in diagnostics:
                st.image(diagnostic["image"], caption=f"{diagnostic['name']} · diagnostic only", width="stretch")
                st.caption(f"Parameters {diagnostic['parameters']} · {diagnostic['processing_ms']:.2f} ms · {diagnostic['limitation']}")


def _render_batch_item(item):
    key = f"batch_{item['index']}_{item['archive_stem']}"
    stages = _photo_stages(item)
    diagnostics = [stage for stage in item["stages"] if stage.get("output_kind") == "diagnostic"]
    st.markdown(f"### {escape(item['filename'])}  <span class='status-pill'>{len(stages)} filters</span>", unsafe_allow_html=True)
    metadata = item.get("source_metadata", {})
    if metadata.get("category"):
        st.caption(f"Category: {metadata['category']} · Source: {metadata.get('source_dataset', 'user upload')}")
    options = ["00 · Original"] + [f"{n:02d} · {s['name']}" for n, s in enumerate(stages, 1)]
    if key + "_stage" not in st.session_state:
        st.session_state[key + "_stage"] = options[1] if len(options) > 1 else options[0]
    current_selection = st.session_state.get(key + "_stage", options[0])
    selected_index = options.index(current_selection) if current_selection in options else 0
    _render_technique_map(item)
    _render_serpentine_timeline(item["original"], stages, key, selected_index, options)
    selected = st.selectbox("Inspect processing stage", options, key=key + "_stage")
    index = options.index(selected)
    current = item["original"] if index == 0 else stages[index - 1]["image"]
    stage = None if index == 0 else stages[index - 1]
    st.progress(index / max(1, len(stages)), text=f"Stage {index + 1} of {len(stages) + 1}" if index else "Stage 1 · Original")
    selected_title = "Stage 1 · Original" if stage is None else f"Stage {index + 1} · {stage['name']}"
    st.markdown(f"### {selected_title}")
    if stage:
        st.markdown(f"**{stage['name']}** · {stage['explanation']}")
        before_after = st.columns(2)
        before_after[0].image(stage["input_image"], caption="Before · previous stage", width="stretch")
        before_after[1].image(current, caption=f"After · {stage['name']}", width="stretch")
        _show_histogram(stage["before"], stage["after"])
    else:
        st.image(current, caption="Unmodified original", width="stretch")
        _metrics_panel(item["input_metrics"], "Original measurements")

    if index == len(stages):
        with st.expander("Original versus final output", expanded=False):
            before_after = st.columns(2)
            before_after[0].image(item["original"], caption="Original", width="stretch")
            before_after[1].image(item["final"], caption="Final output", width="stretch")
    if diagnostics:
        with st.expander("Advanced diagnostics", expanded=False):
            for diagnostic in diagnostics:
                st.image(diagnostic["image"], caption=f"{diagnostic['name']} · diagnostic only", width="stretch")
                st.caption(diagnostic["limitation"])
    st.download_button("Download final PNG", png_bytes(item["final"]),
                       file_name=f"{item['archive_stem']}_final.png", mime="image/png", key=key + "_png")


def _navigate_to(page, mode=None):
    page_aliases = {
        "Command Center": "Traffic Vision Lab",
        "Image Workbench": "Processing Studio",
        "Pipeline & Evaluation": "Results & Evaluation",
        "Research Notes": "Results & Evaluation",
    }
    page = page_aliases.get(page, page)
    st.session_state["app_section"] = page
    if mode is not None:
        st.session_state["workspace_mode"] = mode


def _overview_session_state():
    """Read real cached results; never infer a live camera or detector state."""
    state = st.session_state
    if state.get("_processing_active"):
        run_status = "Processing"
    elif state.get("_batch_results") or any(
            key.startswith("_single_") and key.endswith("_result") for key in state.keys()):
        run_status = "Results available"
    else:
        run_status = "Awaiting input"
    evaluation_available = any(
        ("_batch_visibility_" in key and isinstance(value, tuple))
        or ("_visibility_" in key and isinstance(value, dict) and "per_image" in value)
        for key, value in state.items()
    )
    return {
        "run_status": run_status,
        "selection_policy": state.get("pipeline_selection_mode", "Adaptive"),
        "evaluation_status": "Generated for this session" if evaluation_available else "Available after processing",
    }

# Helper to record processing history entries for session analytics
def _record_history(filename: str, workspace: str, stage_count: int, processing_ms: float):
    hist = st.session_state.setdefault('_history', [])
    hist.append({
        'time': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        'filename': filename,
        'workspace': workspace,
        'stage_count': stage_count,
        'processing_ms': processing_ms,
    })
    """Read real cached results; never infer a live camera or detector state."""
    state = st.session_state
    if state.get("_processing_active"):
        run_status = "Processing"
    elif state.get("_batch_results") or any(
            key.startswith("_single_") and key.endswith("_result") for key in state.keys()):
        run_status = "Results available"
    else:
        run_status = "Awaiting input"
    evaluation_available = any(
        ("_batch_visibility_" in key and isinstance(value, tuple))
        or ("_visibility_" in key and isinstance(value, dict) and "per_image" in value)
        for key, value in state.items()
    )
    return {
        "run_status": run_status,
        "selection_policy": state.get("pipeline_selection_mode", "Adaptive"),
        "evaluation_status": "Generated for this session" if evaluation_available else "Available after processing",
    }


def _cached_visibility_results():
    reports, categories = [], []
    for key, value in st.session_state.items():
        if "_batch_visibility_" in key and isinstance(value, tuple) and len(value) == 2:
            reports.extend(value[0])
            categories.extend(value[1])
        elif "_visibility_" in key and isinstance(value, dict) and "per_image" in value:
            reports.append(value)
    return reports, categories


def _render_command_center():
    state = _overview_session_state()
    st.markdown("""
    <section class="command-hero">
      <div class="section-kicker">SMART TRAFFIC MANAGEMENT · IMAGE INTELLIGENCE</div>
      <h1>Traffic Vision Control Room</h1>
      <p><strong>Adverse-weather image enhancement and evaluation</strong><br>A research workbench for explainable classical digital image processing.</p>
      <p>Inspect traffic imagery, review the rule-based enhancement plan, and compare measured image changes across each processing stage.</p>
    </section>
    """, unsafe_allow_html=True)
    status_cols = st.columns(4)
    values = [
        ("Processing engine", "Classical DIP"),
        ("Selection policy", state["selection_policy"]),
        ("Session", state["run_status"]),
        ("Evaluation", state["evaluation_status"]),
    ]
    for column, (label, value) in zip(status_cols, values):
        column.markdown(f"<div class='status-card'><div class='label'>{label}</div><div class='value'>{escape(str(value))}</div></div>", unsafe_allow_html=True)

    st.markdown("### Open a workspace")
    # Recent processing history
    st.markdown("### Recent processing history")
    history = st.session_state.get('_history', [])
    if history:
        hist_df = []
        for entry in history[-5:][::-1]:
            hist_df.append({
                "Time": entry.get('time'),
                "File": entry.get('filename'),
                "Workspace": entry.get('workspace'),
                "Stages": entry.get('stage_count', '-'),
                "Time (ms)": entry.get('processing_ms', '-')
            })
        st.dataframe(hist_df, hide_index=True, width="stretch")
    else:
        st.info("No processing history yet. Run a single image or batch to populate this table.")
    # Session reset with confirmation
    if st.button("Reset session", help="Clear cached results and history"):
        st.session_state['reset_requested'] = True
    if st.session_state.get('reset_requested'):
        st.warning("Are you sure you want to reset the session? This will clear all cached results and history.")
        col_yes, col_no = st.columns([1, 1])
        if col_yes.button("Confirm reset"):
            keys_to_remove = [k for k in st.session_state.keys() if k.startswith(("_single_", "_batch_")) or k in ("_processing_active", "_history")]
            for k in keys_to_remove:
                st.session_state.pop(k, None)
            st.session_state.pop('reset_requested', None)
            st.success("Session reset.")
            st.rerun()
        if col_no.button("Cancel"):
            st.session_state.pop('reset_requested', None)

    single, batch, methodology = st.columns(3)
    single.button("Start single-image processing", type="primary", use_container_width=True,
                  on_click=_navigate_to, args=("Processing Studio", "Guided single image"), key="overview_single")
    batch.button("Open batch processing", use_container_width=True,
                 on_click=_navigate_to, args=("Processing Studio", "Batch processing"), key="overview_batch")
    methodology.button("Review results & methodology", use_container_width=True,
                       on_click=_navigate_to, args=("Results & Evaluation",), key="overview_methodology")

    st.markdown("### Why traffic imagery degrades")
    conditions = [
        ("Fog and haze", "Veiling light can flatten contrast and soften vehicle boundaries."),
        ("Rain", "Water and streaks can obscure road markings and fine structures; rain-specific removal is not implemented."),
        ("Snow", "Bright particles and low contrast can hide lane edges and scene detail."),
        ("Low illumination", "Weak exposure can reduce visible shadow detail and amplify image noise."),
        ("Noise", "Sensor or compression variation can compete with small vehicle and marking details."),
        ("Motion blur", "Camera or vehicle motion can smear edges; the current pipeline does not estimate blur motion."),
    ]
    for row_start in (0, 3):
        cols = st.columns(3)
        for column, (name, description) in zip(cols, conditions[row_start:row_start + 3]):
            column.markdown(f"<div class='condition-card'><strong>{name}</strong><span>{description}</span></div>", unsafe_allow_html=True)
    st.caption("These are general degradation mechanisms, not image classifications. The adaptive planner uses measured image statistics and documented rules.")

    st.markdown("### DAWN sample imagery")
    st.caption("Original bundled source images, shown as examples of the dataset categories. The category is source metadata, not a model prediction or clean reference.")
    manifest = _load_dawn_manifest()
    selected_rows = []
    for category in ("fog", "rain", "snow", "sandstorm"):
        row = next((item for item in manifest if item.get("category") == category), None)
        if row:
            try:
                selected_rows.append((category, row, _dawn_path(row)))
            except (KeyError, OSError, ValueError):
                continue
    if selected_rows:
        cols = st.columns(len(selected_rows))
        for column, (category, row, path) in zip(cols, selected_rows):
            column.image(str(path), caption=f"DAWN · {category.title()} · {row['filename']}", width="stretch")
    else:
        st.info("Bundled DAWN sample images are not available in this checkout.")

    st.markdown("### What the workbench does")
    st.markdown("""
    The system measures image characteristics, builds a transparent rule-based plan, applies selected operations sequentially, and retains each real intermediate output. Evaluation reports describe measured changes and heuristic indicators; they do not by themselves establish better human-perceived visibility or downstream detection accuracy.
    """)
    _render_technique_map()


def _render_research_evaluation_workbench():
    st.markdown("### Batch Research Evaluation & Statistical Benchmarking")
    st.caption(
        "Execute automated statistical evaluation across adverse-weather categories (fog, rain, snow, sandstorm). "
        "Calculates per-category aggregations (mean, median, standard deviation), processing latency, throughput in megapixels/second, "
        "and paired reference metrics (strictly marking unpaired scenes as unavailable). "
        "Outputs reproducible run manifests and research-grade JSON/CSV exports."
    )

    with st.container(border=True):
        c1, c2, c3 = st.columns([2, 1, 1])
        selected_cats = c1.multiselect(
            "Target Weather Categories",
            options=["fog", "rain", "snow", "sandstorm"],
            default=["fog", "rain", "snow", "sandstorm"],
            key="research_eval_cats",
        )
        sample_count = c2.slider("Images per category", min_value=1, max_value=6, value=2, key="research_eval_count")
        eval_mode = c3.selectbox(
            "Evaluation Pipeline",
            ["Adaptive", "CLAHE only", "Dehazing only", "Normalization only", "Denoising only"],
            key="research_eval_mode",
        )

        run_btn = st.button("Run Statistical Research Benchmark", type="primary", key="btn_run_research_eval")

    if run_btn:
        if not selected_cats:
            st.warning("Please select at least one weather category.")
            return

        manifest_settings = {
            "categories": selected_cats,
            "images_per_category": sample_count,
            "pipeline_mode": eval_mode,
        }
        manifest = create_run_manifest(
            dataset_name="DAWN Adverse Weather Dataset",
            pipeline_mode=eval_mode,
            algorithm_settings=manifest_settings,
            notes="Statistical evaluation of classical DIP enhancement across real adverse-weather conditions.",
        )

        mode_str = "Adaptive" if eval_mode == "Adaptive" else "Fixed order"

        per_image_records = []
        progress_bar = st.progress(0.0)
        status_text_slot = st.empty()

        task_items = []
        for cat in selected_cats:
            cat_dir = ROOT / "data" / "samples" / cat
            if cat_dir.is_dir():
                valid_files = sorted([f for f in cat_dir.iterdir() if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}])
                for fpath in valid_files[:sample_count]:
                    task_items.append((fpath, cat))

        total_tasks = len(task_items)
        if total_tasks == 0:
            st.error("No valid sample images found for selected categories.")
            return

        for i, (fpath, cat) in enumerate(task_items):
            status_text_slot.caption(f"Evaluating ({i+1}/{total_tasks}): `{cat}/{fpath.name}`…")
            try:
                rec = evaluate_single_image(
                    fpath,
                    filename=fpath.name,
                    category=cat,
                    reference_image=None,
                    pipeline_mode=mode_str,
                    max_dim=1280,
                )
                per_image_records.append(rec)
            except Exception as exc:
                st.error(f"Failed to evaluate {fpath.name}: {exc}")
            progress_bar.progress((i + 1) / total_tasks)

        status_text_slot.empty()
        progress_bar.empty()

        cat_summaries = aggregate_category_statistics(per_image_records)
        st.session_state["_research_eval_results"] = {
            "manifest": manifest,
            "per_image": per_image_records,
            "category_summaries": cat_summaries,
        }
        st.success(f"Benchmark completed successfully! Evaluated {len(per_image_records)} images across {len(cat_summaries)} categories.")

    eval_data = st.session_state.get("_research_eval_results")
    if eval_data:
        manifest = eval_data["manifest"]
        per_image = eval_data["per_image"]
        cat_summaries = eval_data["category_summaries"]

        st.markdown("#### Experiment Run Manifest")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Run UUID", manifest["run_id"][:8] + "…")
        m2.metric("System Platform", manifest["platform"]["system"])
        m3.metric("Python Version", manifest["packages"]["python"])
        m4.metric("Streamlit Version", manifest["packages"]["streamlit"])

        st.markdown("#### Category Statistical Aggregations")
        cat_rows = []
        for cat, s in cat_summaries.items():
            cat_rows.append({
                "Category": cat.title(),
                "Images Evaluated": s["image_count"],
                "Mean Latency (ms)": f"{s['latency_ms']['mean']:.2f}" if s['latency_ms']['mean'] is not None else "—",
                "Median Latency (ms)": f"{s['latency_ms']['median']:.2f}" if s['latency_ms']['median'] is not None else "—",
                "Std Dev Latency (ms)": f"{s['latency_ms']['std_dev']:.2f}" if s['latency_ms']['std_dev'] is not None else "—",
                "Throughput (MP/s)": f"{s['throughput_mp_per_sec']['mean']:.2f}" if s['throughput_mp_per_sec']['mean'] is not None else "—",
                "Mean Δ Contrast (σ)": f"{s['delta_contrast']['mean']:+.2f}" if s['delta_contrast']['mean'] is not None else "—",
                "Mean Δ Sharpness": f"{s['delta_sharpness']['mean']:+.1f}" if s['delta_sharpness']['mean'] is not None else "—",
                "Mean Δ Noise Proxy": f"{s['delta_noise_proxy']['mean']:+.2f}" if s['delta_noise_proxy']['mean'] is not None else "—",
            })
        st.dataframe(cat_rows, hide_index=True, width="stretch")

        st.markdown("#### Comparative Metric Visualizations")
        bc1, bc2 = st.columns(2)
        with bc1:
            st.caption("Mean Execution Latency by Weather Category (ms) — lower is faster")
            lat_chart = {cat.title(): s['latency_ms']['mean'] for cat, s in cat_summaries.items() if s['latency_ms']['mean'] is not None}
            st.bar_chart(lat_chart, height=220)
        with bc2:
            st.caption("Mean Contrast Gain (Δ σ) — positive indicates restored dynamic range")
            contrast_chart = {cat.title(): s['delta_contrast']['mean'] for cat, s in cat_summaries.items() if s['delta_contrast']['mean'] is not None}
            st.bar_chart(contrast_chart, height=220)

        with st.expander("Detailed Per-Image Metric Records", expanded=False):
            img_table = []
            for rec in per_image:
                om = rec.get("original_metrics", {})
                fm = rec.get("final_metrics", {})
                deltas = rec.get("deltas", {})
                ref = rec.get("reference_metrics", {})
                img_table.append({
                    "Filename": rec["filename"],
                    "Category": rec["category"],
                    "Resolution": f"{rec['width']}x{rec['height']}",
                    "Latency (ms)": f"{rec['processing_ms']:.2f}",
                    "Throughput (MP/s)": f"{rec['throughput_mp_per_sec']:.2f}",
                    "Stages": rec["stage_count"],
                    "Δ Brightness": f"{deltas.get('delta_brightness', 0.0):+.2f}",
                    "Δ Contrast": f"{deltas.get('delta_contrast', 0.0):+.2f}",
                    "Δ Sharpness": f"{deltas.get('delta_sharpness', 0.0):+.1f}",
                    "Δ Noise Proxy": f"{deltas.get('delta_noise_proxy', 0.0):+.2f}",
                    "PSNR / SSIM": "Unavailable (unpaired)" if ref.get("psnr") is None else f"{ref['psnr']:.2f}dB / {ref['ssim']:.4f}",
                })
            st.dataframe(img_table, hide_index=True, width="stretch")

        st.caption("Research Integrity Note: PSNR and SSIM require perfectly registered pixel-level clear-weather reference pairs. Real-world adverse weather traffic surveillance imagery is unpaired; substituting arbitrary clear scenes or synthesizing pseudo-ground-truth produces invalid science. In this evaluation, reference metrics are explicitly marked as unavailable.")

        st.markdown("#### Export Scientific Experiment Artifacts")
        exp1, exp2, exp3 = st.columns(3)
        exp1.download_button(
            "Download Research Report (JSON)",
            research_report_json_bytes(manifest, per_image, cat_summaries),
            file_name=f"research_report_{manifest['run_id'][:8]}.json",
            mime="application/json",
            key="dl_research_json",
        )
        exp2.download_button(
            "Download Category Summary (CSV)",
            category_summary_csv_bytes(cat_summaries),
            file_name=f"category_summary_{manifest['run_id'][:8]}.csv",
            mime="text/csv",
            key="dl_category_csv",
        )
        exp3.download_button(
            "Download Per-Image Records (CSV)",
            per_image_results_csv_bytes(per_image),
            file_name=f"per_image_metrics_{manifest['run_id'][:8]}.csv",
            mime="text/csv",
            key="dl_per_image_csv",
        )


def _render_pipeline_and_evaluation():
    st.markdown("<div class='section-kicker'>Method and evidence</div>", unsafe_allow_html=True)
    st.title("Results & Evaluation")
    st.caption("A reproducible, rule-based classical DIP workflow. Weather labels are provenance metadata; no trained degradation classifier is used.")
    pipeline_steps = [
        ("01 · Input", "Validated upload or bundled DAWN sample."),
        ("02 · Analyze", "Luminance, contrast, edges, clipping, noise and haze-like indicators."),
        ("03 · Select", "Rules choose methods from measured evidence and user settings."),
        ("04 · Process", "Each operation consumes the previous stage's actual output."),
        ("05 · Evaluate", "Compare image statistics and valid DAWN vehicle regions where available."),
        ("06 · Downstream", "Incident analysis is planned; it is not integrated."),
    ]
    for row_start in (0, 3):
        cols = st.columns(3)
        for column, (name, description) in zip(cols, pipeline_steps[row_start:row_start + 3]):
            column.markdown(f"<div class='traffic-step'><div class='number'>{name}</div><div class='detail'>{description}</div></div>", unsafe_allow_html=True)
        if row_start == 0:
            st.caption("Input  →  analysis  →  adaptive technique selection  ↓")
        else:
            st.caption("Sequential enhancement  →  measured visibility changes  →  future incident study")

    implemented, planned = st.columns(2)
    with implemented.container(border=True):
        st.markdown("### <span class='implemented-tag'>Implemented</span>", unsafe_allow_html=True)
        st.markdown("""
        - Rule-based adaptive planning and user-selected fixed order.
        - Sequential classical enhancement with saved stage outputs.
        - Stage inspection, previous/current comparisons and diagnostics.
        - Image-quality measurements and visibility indicators.
        - DAWN VOC vehicle-region comparisons when matching annotations exist.
        - Per-image, per-stage, region and category CSV reports; batch ZIP export.
        - Optional general COCO vehicle counts, only when local weights and dependencies are available.
        """)
    with planned.container(border=True):
        st.markdown("### <span class='planned-tag'>Planned / not validated</span>", unsafe_allow_html=True)
        st.markdown("""
        - Accident and blocked-lane detection.
        - Downstream comparison against suitable incident ground truth.
        - Validation that enhancement improves detection under a documented protocol.
        - A trained degradation classifier, if future data and evaluation justify one.
        """)

    st.markdown("### Visibility evaluation from this session")
    st.caption("Metrics and heuristic indicators describe measurable image changes. They do not, by themselves, prove improved human-perceived visibility or incident-detection accuracy.")
    reports, categories = _cached_visibility_results()
    if reports:
        rows = []
        for report in reports:
            image = report["per_image"]
            rows.append({
                "Image": image["filename"], "Weather metadata": image["category"],
                "Stages": image["stage_count"], "Time (ms)": image["total_processing_ms"],
                "Mean luminance Δ": image["original_to_final_metric_differences"]["mean_luminance"]["absolute"],
                "Contrast σ Δ": image["original_to_final_metric_differences"]["luminance_stddev"]["absolute"],
                "Vehicle regions": image["vehicle_region_count"],
                "Annotation status": image["vehicle_annotation_status"],
            })
        st.dataframe(rows, hide_index=True, width="stretch")
        if categories:
            st.markdown("**Descriptive weather-category aggregates**")
            st.dataframe(categories, hide_index=True, width="stretch")
        st.caption("Vehicle-region statistics are not detection accuracy. Noise and haze values are proxies; a metric increase is not a universal quality improvement.")
        st.markdown("**Download session measurements**")
        export_cols = st.columns(4)
        exports = (
            ("Per-image CSV", visibility_image_csv_bytes(reports), "traffic_visibility_per_image.csv"),
            ("Per-stage CSV", visibility_stage_csv_bytes(reports), "traffic_visibility_per_stage.csv"),
            ("Vehicle regions CSV", visibility_region_csv_bytes(reports), "traffic_visibility_vehicle_regions.csv"),
            ("Category summary CSV", visibility_category_csv_bytes(categories), "traffic_visibility_by_category.csv"),
        )
        for column, (label, payload, filename) in zip(export_cols, exports):
            column.download_button(label, payload, file_name=filename, mime="text/csv",
                                   key="session_export_" + filename)
    else:
        st.info("No visibility evaluation has been generated in this session yet. Process an image in the workbench to populate measured results.")

    st.markdown("### Interpretation guide")
    measured, heuristic, ground_truth = st.columns(3)
    measured.markdown("**Measured values**\n\nPixel-derived luminance, contrast, clipping, edge response, Laplacian variance, changed pixels and elapsed time.")
    heuristic.markdown("**Heuristic proxies**\n\nNoise residual and haze-like indicators are sensitive to texture, lighting and scene content; they are not weather probabilities.")
    ground_truth.markdown("**Ground-truth outcomes**\n\nDAWN vehicle boxes support matched-region measurements only. No aligned clear references or plate annotations are supplied.")
    st.divider()
    _render_research_evaluation_workbench()
    st.divider()

    with st.expander("Research scope and limitations"):
        st.markdown("""
        **Aim.** Develop an adaptive image-processing system for degraded traffic-camera imagery, then investigate whether measured changes translate into improved interpretation when suitable downstream ground truth becomes available.

        The selector is a deterministic, rule-based policy, not a trained weather classifier. Noise and haze measurements are heuristic proxies. DAWN VOC annotations support vehicle-region comparisons for matching images; they do not provide plate ground truth. The available images are not aligned clear-weather pairs, so paired PSNR/SSIM is omitted. Vehicle-region statistics do not establish detection accuracy. Accident and blocked-lane detection and downstream validation are planned, not integrated.
        """)
        st.markdown("**Future validation:** enhanced traffic image → incident detection → ground-truth comparison → downstream evaluation. Suitable annotations and a documented protocol are required before reporting detector performance.")


def _render_research_notes():
    st.markdown("<div class='section-kicker'>Research scope</div>", unsafe_allow_html=True)
    st.title("Research notes & limitations")
    st.markdown("""
    **Aim.** Develop an adaptive image-processing system for degraded traffic-camera imagery, then investigate whether measured changes translate to improved interpretation when suitable downstream ground truth becomes available.

    **Current contribution.** A deterministic rule-based selector chooses from classical spatial, luminance, dehazing and frequency-domain methods. The pipeline records the selected sequence and actual intermediate images. It is not a trained weather classifier.
    """)
    limits = [
        ("Image statistics", "Brightness, contrast, sharpness and clipping summarize pixel properties. Higher contrast or sharpness is not proof of better visibility."),
        ("Noise and haze", "Residual noise and haze-like values are heuristic proxies, not true noise-removal rates or verified weather labels."),
        ("DAWN annotations", "VOC vehicle boxes enable region comparisons for matched bundled images. They do not provide license-plate ground truth."),
        ("Reference metrics", "The available images are not aligned clear-weather pairs, so PSNR and SSIM are omitted."),
        ("Traffic analysis", "Optional COCO vehicle counts are not ground-truth accuracy. Accident and blocked-lane detection are not integrated."),
    ]
    for start in (0, 3):
        cols = st.columns(3 if start == 0 else 2)
        for column, (name, detail) in zip(cols, limits[start:start + (3 if start == 0 else 2)]):
            with column.container(border=True):
                st.markdown(f"**{name}**")
                st.write(detail)
    st.markdown("### Future validation sequence")
    st.markdown("""
    **Enhanced traffic image → incident detection → ground-truth comparison → downstream evaluation**

    Planned categories include traffic accidents and blocked lanes. A future study needs suitable annotations and a fixed evaluation protocol to compare original and enhanced inputs. No incident output or accuracy claim is generated by this application today.
    """)
    with st.expander("Dataset provenance and evaluation notes"):
        st.markdown("DAWN category labels are retained as source metadata. Bundled sample paths and attribution are listed in `data/metadata/dawn_sample_manifest.csv`. The local VOC XML annotations are used for vehicle-region measurements only. See the README for the dataset source, license notes, coordinate handling, and benchmark limitations.")


def main():
    st.set_page_config(page_title="Smart Traffic Management", page_icon="◈", layout="wide", initial_sidebar_state="expanded")
    _inject_css()
    old_pages = {"Command Center": "Traffic Vision Lab", "Image Workbench": "Processing Studio",
                 "Pipeline & Evaluation": "Results & Evaluation", "Research Notes": "Results & Evaluation"}
    saved_page = st.session_state.get("app_section")
    if saved_page in old_pages:
        st.session_state["app_section"] = old_pages[saved_page]
    reset_nonce = st.session_state.get("reset_counter", 0)
    nav_identity, nav_page, nav_reset = st.columns([1.1, 2.8, .55])
    with nav_identity:
        st.markdown("<div class='brand-row'><div class='brand-mark'>TV</div><div><div class='brand-title'>SMART TRAFFIC</div><div class='brand-sub'>Image intelligence workbench</div></div></div>", unsafe_allow_html=True)
    with nav_page:
        with st.container(key="section-nav"):
            page = st.radio("Application section", ["Traffic Vision Lab", "Processing Studio",
                                                     "Results & Evaluation"],
                            horizontal=True, key="app_section", label_visibility="collapsed")
    with nav_reset:
        if st.button("Reset", help="Clear cached single-image and batch processing results"):
            st.session_state["reset_counter"] = reset_nonce + 1
            for key in list(st.session_state.keys()):
                if key.startswith(("_single_", "_batch_")) or key == "_processing_active":
                    st.session_state.pop(key, None)
            st.rerun()

    if page == "Traffic Vision Lab":
        _render_command_center()
        return
    if page == "Results & Evaluation":
        _render_pipeline_and_evaluation()
        return

    mode = st.radio("Processing workspace", ["Guided single image", "Algorithm Benchmark Matrix", "Batch processing"], horizontal=True,
                    key="workspace_mode", label_visibility="collapsed")
    with st.sidebar:
        st.markdown("<div class='eyebrow'>Project controls</div>", unsafe_allow_html=True)
        st.caption("Settings are applied to the next run; cached outputs remain available until settings or source change.")
        st.divider()
        pipeline_mode = st.radio("Pipeline selection", ["Adaptive", "Fixed order"],
                                 help="Adaptive is the default rule-based plan. Fixed order runs only the methods you select, in the displayed selection order.",
                                 key="pipeline_selection_mode")
        selected_operations = None
        if pipeline_mode == "Fixed order":
            st.caption("Choose concrete existing enhancement methods. Processing order follows this list; denoising uses the method below.")
            with st.expander("Method library and purposes", expanded=False):
                for operation, purpose in OPERATION_PURPOSES.items():
                    st.markdown(f"**{operation}.** {purpose}")
            selected_operations = st.multiselect("Ordered operations", PIPELINE_OPERATIONS,
                                                 key="fixed_operations")
        st.markdown("**Adaptive options**")
        denoise = st.selectbox("Noise handling", ["automatic", "off", "median", "gaussian", "bilateral", "frequency"],
                               help="Automatic uses the measured noise proxy to decide whether a mild FFT low-pass is justified; spatial filters are available for explicit selection.")
        dehaze = st.selectbox("Dark-channel dehazing", ["automatic", "off", "force"],
                              help="Automatic uses conservative image statistics. Force is experimental and may distort non-hazy images.")
        edge_diagnostic = st.checkbox("Include Sobel diagnostic output", value=False)
        morphology_choice = st.selectbox("Morphology diagnostic", ["off", "erosion", "dilation", "opening", "closing", "gradient", "tophat", "blackhat"],
                                         help="Produces an auxiliary grayscale diagnostic, not a final photograph.")
        
        with st.expander("Interactive Algorithm Tuning", expanded=False):
            st.caption("Fine-tune algorithm hyperparameters. Defaults follow adaptive rules.")
            clahe_clip = st.slider("CLAHE Clip Limit", 1.0, 5.0, 2.0, 0.25, key="p_clahe_clip")
            clahe_tile = st.selectbox("CLAHE Tile Size", [16, 32, 64, 128], index=2, key="p_clahe_tile")
            dehaze_omega = st.slider("Dehazing Omega", 0.40, 0.95, 0.70, 0.05, key="p_dehaze_omega")
            dehaze_floor = st.slider("Dehazing Transmission Floor", 0.20, 0.75, 0.55, 0.05, key="p_dehaze_floor")
            unsharp_radius = st.slider("Unsharp Mask Radius", 0.5, 3.0, 1.2, 0.1, key="p_unsharp_rad")
            unsharp_percent = st.slider("Unsharp Mask Percent (%)", 50, 250, 90, 10, key="p_unsharp_pct")
            norm_low = st.slider("Percentile Stretch Low (%)", 0.0, 5.0, 1.0, 0.5, key="p_norm_low")
            norm_high = st.slider("Percentile Stretch High (%)", 95.0, 100.0, 99.0, 0.5, key="p_norm_high")
            max_res = st.selectbox("Resolution Constraint", ["Original resolution", "1920px (1080p limit)", "1280px (720p limit)", "800px"], index=0, key="p_max_res")
            res_map = {"Original resolution": None, "1920px (1080p limit)": 1920, "1280px (720p limit)": 1280, "800px": 800}
            max_dim_val = res_map[max_res]
            custom_params = {
                "CLAHE on luminance": {"clip_limit": clahe_clip, "tile_size": clahe_tile},
                "Dark-channel-prior dehazing": {"omega": dehaze_omega, "transmission_floor": dehaze_floor},
                "Luminance unsharp mask": {"radius": unsharp_radius, "percent": unsharp_percent},
                "Robust luminance normalization": {"low_percentile": norm_low, "high_percentile": norm_high},
            }

        st.caption(f"Up to {MAX_BATCH_IMAGES} batch images · {MAX_IMAGE_PIXELS//1_000_000} MP each")
        st.divider()
        st.caption("Rule-based adaptive DIP · no trained weather classifier")

    st.markdown("<div class='section-kicker'>Processing studio · classical image processing</div>", unsafe_allow_html=True)
    st.markdown("### Process degraded traffic imagery")
    st.caption("Select a single image or batch mode. The sidebar controls the rule-based adaptive plan or your fixed operation sequence.")
    st.markdown("<div style='height:1px;background:#2a3039;margin:12px 0 18px'></div>", unsafe_allow_html=True)

    # Helper to record processing history entries
    def _record_history(filename: str, workspace: str, stage_count: int, processing_ms: float):
        hist = st.session_state.setdefault('_history', [])
        hist.append({
            'time': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            'filename': filename,
            'workspace': workspace,
            'stage_count': stage_count,
            'processing_ms': processing_ms,
        })

    if mode == "Guided single image":
        top_left, top_right = st.columns([3, 1])
        top_left.markdown("#### Single-image study")
        top_right.markdown("<span class='status-pill'>Analysis studio</span>", unsafe_allow_html=True)
        
        single_source = st.radio("Image source", ["Select a bundled DAWN sample", "Upload an image"],
                                 horizontal=True, key=f"single_src_toggle_{reset_nonce}")
        
        data = None
        filename = ""
        metadata = {}
        annotation = None
        
        if single_source == "Select a bundled DAWN sample":
            manifest = _load_dawn_manifest()
            categories = sorted({row["category"] for row in manifest})
            c_col, f_col = st.columns([1, 2])
            chosen_cat = c_col.selectbox("DAWN weather category", categories, key=f"single_dawn_cat_{reset_nonce}")
            rows = [r for r in manifest if r["category"] == chosen_cat]
            chosen_file = f_col.selectbox("Sample image", [r["filename"] for r in rows], key=f"single_dawn_file_{reset_nonce}")
            row = next(r for r in rows if r["filename"] == chosen_file)
            try:
                path = _dawn_path(row)
                data = path.read_bytes()
                filename = chosen_file
                metadata = {"source_dataset": "DAWN", "category": chosen_cat, "source_mode": None}
                annotation = find_dawn_annotation(_load_dawn_annotation_index(), filename, chosen_cat)
            except Exception as exc:
                st.error(f"Could not load DAWN sample: {exc}")
        else:
            weather = st.selectbox("Weather category (optional metadata)", CATEGORIES, key=f"single_weather_{reset_nonce}")
            upload = st.file_uploader("Drop a traffic image here or browse", type=SUPPORTED, key=f"single_file_{reset_nonce}")
            if upload:
                data = upload.getvalue()
                filename = upload.name
                metadata = {"source_dataset": "user-upload", "category": "" if weather == "unspecified" else weather,
                            "source_mode": None}
        
        if not data:
            st.markdown("<div class='empty-card'><div class='eyebrow'>Analysis workspace</div><h3>Start with an original image</h3><p>Choose a bundled DAWN sample or upload a traffic image. The application will measure its image statistics, construct a reproducible plan, and show each real processing stage beside the unchanged original.</p><p style='color:#9ba5b1'>PNG · JPEG · BMP · TIFF · WebP · 16 megapixels maximum</p></div>", unsafe_allow_html=True)
        else:
            state_key = "_single_" + sha256((filename + repr(metadata) + str(reset_nonce)).encode()).hexdigest()[:12]
            options = repr((denoise, dehaze, edge_diagnostic, morphology_choice,
                            pipeline_mode, selected_operations, custom_params, max_dim_val)).encode()
            digest = sha256(data + repr(sorted(metadata.items())).encode() + options).hexdigest()
            cached = st.session_state.get(state_key + "_digest") == digest
            start_processing = st.button("Start DIP Processing", type="primary", key=state_key + "_start")
            if pipeline_mode == "Fixed order" and not selected_operations:
                st.info("Select at least one method in the ordered pipeline, or switch to Adaptive.")
            elif start_processing or cached:
                try:
                    progress_slot = st.empty()
                    callback = _make_progress_callback(progress_slot) if start_processing and not cached else None
                    with st.container():
                        result = _get_pipeline(data, filename, metadata, denoise, dehaze, state_key,
                                               edge_diagnostic, morphology_choice, selected_operations,
                                               callback, pipeline_mode, custom_parameters=custom_params,
                                               max_dim=max_dim_val)
                except (ValueError, OSError, MemoryError) as exc:
                    st.error(f"Could not analyze this image: {exc}")
                else:
                    st.caption("The completed run is saved. The timeline below navigates the retained outputs without rerunning or mutating the source image.")
                    c1, c2 = st.columns([3, 1])
                    c1.markdown(f"#### {escape(filename)}")
                    c2.markdown(f"<span class='status-pill'>{_status_text(result)}</span>", unsafe_allow_html=True)
                    _render_result(result, filename, state_key)
                    _render_single_vehicle_analysis(result, state_key, annotation=annotation)
                    with st.expander("FFT frequency-spectrum visualization"):
                        st.image(spectrum_image(result["original"]), caption="Centered log-magnitude 2D FFT of original luminance", width="stretch")
            else:
                st.info("Image loaded and validated. Select the adaptive options or parameter tuning in the sidebar, then click **Start DIP Processing**.")
    elif mode == "Algorithm Benchmark Matrix":
        _render_algorithm_benchmark(reset_nonce, custom_params=custom_params)
    else:
        st.markdown("#### Batch processing")
        source_mode = st.radio("Image source", ["Upload images", "Bundled DAWN samples"], horizontal=True, key=f"batch_source_{reset_nonce}")
        items = []
        errors = []
        if source_mode == "Upload images":
            category = st.selectbox("Weather category for these uploads (optional metadata)", CATEGORIES, key=f"batch_weather_{reset_nonce}")
            uploads = st.file_uploader("Add up to 10 images", type=SUPPORTED, accept_multiple_files=True, key=f"batch_files_{reset_nonce}")
            if uploads:
                if len(uploads) > MAX_BATCH_IMAGES:
                    st.error(f"Select no more than {MAX_BATCH_IMAGES} images at once.")
                else:
                    for file in uploads:
                        items.append((file.name, file.getvalue(), {"source_dataset": "user-upload",
                                      "category": "" if category == "unspecified" else category}))
        else:
            manifest = _load_dawn_manifest()
            categories = sorted({row["category"] for row in manifest})
            if not manifest:
                st.error("The DAWN image manifest is unavailable.")
            else:
                category = st.selectbox("DAWN weather category", categories, key=f"dawn_category_{reset_nonce}")
                rows = [row for row in manifest if row["category"] == category]
                names = st.multiselect("Select up to 10 bundled DAWN images", [r["filename"] for r in rows],
                                       max_selections=MAX_BATCH_IMAGES, key=f"dawn_files_{reset_nonce}")
                lookup = {r["filename"]: r for r in rows}
                for name in names:
                    row = lookup[name]
                    try:
                        path = _dawn_path(row)
                        meta = {"source_dataset": "DAWN", "category": category,
                                "source_url": row.get("license_source", ""),
                                "license": "CC BY-NC 3.0", "attribution": "Mourad A. Kenk; DAWN dataset"}
                        items.append((name, path.read_bytes(), meta))
                    except (OSError, ValueError) as exc:
                        errors.append({"filename": name, "error": str(exc)})
                st.caption("DAWN labels are retained as source metadata only; they are not used as a weather classifier or enhancement ground truth.")
        for name, error in errors:
            st.error(f"Could not read {name}: {error}")
        if items and pipeline_mode == "Fixed order" and not selected_operations:
            st.info("Select at least one method in the ordered pipeline, or switch to Adaptive.")
        elif items:
            signature = sha256(b"".join(name.encode() + data + repr(meta).encode() for name, data, meta in items)
                               + denoise.encode() + dehaze.encode() + str(edge_diagnostic).encode()
                               + morphology_choice.encode() + repr((pipeline_mode, selected_operations)).encode()
                               + str(reset_nonce).encode()).hexdigest()
            if st.session_state.get("_batch_digest") != signature:
                try:
                    batch_progress_slot = st.empty()
                    callback = _make_batch_progress_callback(batch_progress_slot)
                    with st.container():
                        st.session_state["_batch_results"] = process_batch(items, denoise_choice=denoise,
                                                                             dehaze_choice=dehaze,
                                                                             edge_diagnostic=edge_diagnostic,
                                                                             morphology_choice=morphology_choice,
                                                                             selected_operations=selected_operations,
                                                                             progress_callback=callback)
                    st.session_state["_batch_digest"] = signature
                except (ValueError, OSError, MemoryError) as exc:
                    st.error(f"Batch processing failed: {exc}")
                    st.session_state.pop("_batch_results", None)
            results = st.session_state.get("_batch_results", [])
            failures = errors + getattr(results, "errors", [])
            for failure in failures:
                st.error(f"Skipped {failure['filename']}: {failure['error']}")
            if results:
                st.success(f"{len(results)} images processed · {len(failures)} skipped · actual stages, reports and originals retained in memory")
                has_dawn_sources = any(str(item.get("source_metadata", {}).get("source_dataset", "")).lower() == "dawn"
                                        for item in results)
                annotation_index = _load_dawn_annotation_index() if has_dawn_sources else {}
                batch_visibility_key = "_batch_visibility_" + str(st.session_state.get("_batch_digest", ""))
                batch_visibility = st.session_state.get(batch_visibility_key)
                if batch_visibility is None:
                    batch_visibility = visibility_reports(results, annotation_index)
                    st.session_state[batch_visibility_key] = batch_visibility
                visibility_items, category_rows = batch_visibility
                with st.expander("Visibility evaluation · batch measurements", expanded=False):
                    st.caption("Category values are descriptive means of image-level changes. Vehicle-region results use matching DAWN VOC boxes only. No plate annotations, paired clear references, or universal visibility score are available.")
                    if category_rows:
                        st.dataframe(category_rows, hide_index=True, width="stretch")
                    v1, v2, v3, v4 = st.columns(4)
                    v1.download_button("Per-image CSV", visibility_image_csv_bytes(visibility_items),
                                       file_name="smart_traffic_visibility_per_image.csv", mime="text/csv",
                                       key=batch_visibility_key + "_image_csv")
                    v2.download_button("Per-stage CSV", visibility_stage_csv_bytes(visibility_items),
                                       file_name="smart_traffic_visibility_per_stage.csv", mime="text/csv",
                                       key=batch_visibility_key + "_stage_csv")
                    v3.download_button("Vehicle regions CSV", visibility_region_csv_bytes(visibility_items),
                                       file_name="smart_traffic_visibility_vehicle_regions.csv", mime="text/csv",
                                       key=batch_visibility_key + "_region_csv")
                    v4.download_button("Category summary CSV", visibility_category_csv_bytes(category_rows),
                                       file_name="smart_traffic_visibility_by_category.csv", mime="text/csv",
                                       key=batch_visibility_key + "_category_csv")
                for item in results:
                    with st.expander(f"{item['filename']} · {len(item['stages'])} executed stage(s)", expanded=False):
                        _render_batch_item(item)
                        with st.expander("Decision log and measurements"):
                            _timeline(item)
                _render_batch_vehicle_analysis(results, signature)
                st.download_button("Download batch ZIP · originals, intermediates, reports, CSV and manifest",
                                   batch_zip(results, failures=failures,
                                             annotation_index=annotation_index,
                                             visibility_evaluation=batch_visibility),
                                   file_name="smart_traffic_dip_batch.zip",
                                   mime="application/zip", key="batch_zip")
                st.download_button("Download batch processing table · CSV", batch_csv_bytes(results),
                                   file_name="smart_traffic_dip_batch.csv", mime="text/csv", key="batch_csv")
            elif failures:
                st.warning("No valid images were processed. Correct the listed files and try again.")
        elif source_mode == "Upload images" and not st.session_state.get("_batch_results"):
            st.markdown("<div class='empty-card'><div class='eyebrow'>Batch workspace</div><h3>Compare a weather set consistently</h3><p>Choose a group of images or browse the bundled DAWN categories. Each file gets its own plan and report; matching category labels are stored as metadata, not treated as ground truth for enhancement quality.</p></div>", unsafe_allow_html=True)
    st.markdown("<div style='height:1px;background:#2a3039;margin:28px 0 12px'></div><div style='color:#89939f;font-size:.76rem'>Classical image processing · deterministic rules · no detector accuracy claims · no paired-reference metrics</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
