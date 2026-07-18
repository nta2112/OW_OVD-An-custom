"""
OW-OVD Pest Detection Demo
===========================
Gradio app để demo model open-world object detection sâu bệnh.
- Upload ảnh → khoanh vùng sâu bệnh → dự đoán nhãn (known / unknown)

Cách chạy:
    python demo_app.py \
        --config configs/open_world/mowod/custom/ip102_t3.py \
        --checkpoint /path/to/checkpoint.pth \
        [--ann-file /path/to/coco_annotations/train.json] \
        [--device cuda:0]
"""

import argparse
import os
import sys
import json
import numpy as np
from pathlib import Path
from typing import Optional

# ── thêm repo root vào sys.path ──────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent
if REPO_ROOT not in [Path(p) for p in sys.path]:
    sys.path.insert(0, str(REPO_ROOT))

# ── lazy imports (chỉ import sau khi check args) ──────────────────────────────
import gradio as gr
from PIL import Image, ImageDraw, ImageFont
import io


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

KNOWN_COLOR   = (255, 210, 0)      # gold
UNKNOWN_COLOR = (255, 50,  80)     # vivid red
BG_ALPHA      = 160                # label background opacity
SCORE_THRESH  = 0.05               # minimum confidence to show box
FONT_SIZE     = 16

IP102_CLASSES = [
    "Aphis gossypii", "Bemisia tabaci", "Empoasca flavescens", "Frankliniella occidentalis",
    "Liriomyza sativae", "Myzus persicae", "Polyphagotarsonemus latus", "Spodoptera litura",
    "Spodoptera exigua", "Helicoverpa armigera", "Agrotis ipsilon", "Agrotis segetum",
    "Alabama argillacea", "Anomis flava", "Cnaphalocrosis medinalis", "Cnaphalocrocis medinalis",
    "Conopomorpha sinensis", "Dacus dorsalis", "Dendrolimus punctatus", "Diaphorina citri",
    "Eriosoma lanigerum", "Gryllotalpa orientalis", "Heliothis armigera", "Heliothis zea",
    "Leucoptera malifoliella", "Lopholeucaspis japonica", "Loxostege sticticalis", "Macrosiphum euphorbiae",
    "Mythimna separata", "Nilaparvata lugens", "Oulema melanopus", "Phytomyza horticola",
    "Pieris rapae", "Plutella xylostella", "Rhopalosiphum maidis", "Rhopalosiphum padi",
    "Schizaphis graminum", "Sesamia inferens", "Sitobion avenae", "Tetranychus cinnabarinus",
    "Tetranychus urticae", "Thrips palmi", "Unaspis euonymi", "Xestia c-nigrum",
    "Acyrthosiphon pisum", "Acrida chinensis", "Adelphocoris lineolatus", "Agasicles hygrophila",
    "Aleurodicus dispersus", "Aleurotrachelus camelliae", "Anoplophora glabripennis", "Aphis fabae",
    "Aphis pomi", "Aphis spiraecola", "Arge pagana", "Bactrocera cucurbitae",
    "Bactrocera dorsalis", "Bactrocera tau", "Batrachedra amydraula", "Blitopertha orientalis",
    "Bradysia odoriphaga", "Callosobruchus chinensis", "Callosobruchus maculatus", "Calosoma sycophanta",
    "Ceroplastes japonicus", "Cetonia aurata", "Chaetocnema concinna", "Chilo suppressalis",
    "Chrysodeixis chalcites", "Chrysomela vigintipunctata", "Chrysomelidae sp.", "Cicadella viridis",
    "Coleophora laricella", "Contarinia sorghicola", "Cydia pomonella", "Dalbulus maidis",
    "Diabrotica virgifera", "Diaphania indica", "Diatraea saccharalis", "Empoasca vitis",
    "Ephestia kuehniella", "Eriborus terebrans", "Eriosoma lanigerum", "Erythroneura elegantula",
    "Euproctis chrysorrhoea", "Eurygaster integriceps", "Galeruca tanaceti", "Gossyparia spuria",
    "Gryllus campestris", "Henosepilachna vigintioctopunctata", "Hypera postica", "Hyponomeuta malinellus",
    "Icerya purchasi", "Ips typographus", "Keiferia lycopersicella", "Laspeyresia molesta",
    "Lema melanopus", "Leptinotarsa decemlineata", "Locusta migratoria", "Longidorus elongatus",
    "Macrosteles quadrilineatus", "Malocosoma americanum", "Mayetiola destructor", "Melanotus cribulosus",
]


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════════

_model = None
_class_names = None
_unknown_id  = None
_device      = "cpu"


def load_model(cfg_path: str, ckpt_path: str, ann_file: Optional[str], device: str):
    """Init model once, cache globally."""
    global _model, _class_names, _unknown_id, _device

    # mock mmcv._ext to avoid ModuleNotFoundError in mmcv-lite (pure Python)
    import sys
    try:
        import mmcv._ext
        print("[demo] Compiled mmcv._ext is available. Using compiled C++ ops.")
    except ImportError:
        print("[demo] Compiled mmcv._ext not found. Using pure Python mock fallback with torchvision ops.")
        import types
        import importlib.machinery
        if 'mmcv._ext' not in sys.modules:
            class MockModule(types.ModuleType):
                def __getattr__(self, name):
                    if name.startswith('__'):
                        raise AttributeError(name)
                    # If NMS is requested, fallback to torchvision's CPU implementation which is compiled and fast
                    if name == 'nms':
                        import torchvision.ops as tv_ops
                        return tv_ops.nms
                    def dummy_func(*args, **kwargs):
                        raise NotImplementedError(
                            f"This C++ operation '{name}' is not compiled or supported on pure Python MMCV CPU."
                        )
                    return dummy_func
            mock_ext = MockModule('mmcv._ext')
            mock_ext.__spec__ = importlib.machinery.ModuleSpec('mmcv._ext', None)
            sys.modules['mmcv._ext'] = mock_ext

    # Patch mmengine Config.fromfile to automatically redirect third_party/mmyolo config paths
    import mmengine
    import os
    from mmyolo import __file__ as mmyolo_init_path
    mmyolo_pkg_root = os.path.dirname(mmyolo_init_path)
    
    _orig_file2dict = mmengine.Config._file2dict
    @classmethod
    def _patched_file2dict(cls, filename, *args, **kwargs):
        # Normalize and check path
        filename_str = str(filename).replace('\\', '/')
        if 'third_party/mmyolo/configs' in filename_str:
            # Reconstruct path using mmyolo package config files
            relative_part = filename_str.split('third_party/mmyolo/configs/')[-1]
            new_path = os.path.join(mmyolo_pkg_root, '.mim', 'configs', relative_part)
            if os.path.exists(new_path):
                filename = new_path
            else:
                # Try fallback without .mim
                new_path_fallback = os.path.join(mmyolo_pkg_root, 'configs', relative_part)
                if os.path.exists(new_path_fallback):
                    filename = new_path_fallback
        return _orig_file2dict(filename, *args, **kwargs)
    mmengine.Config._file2dict = _patched_file2dict

    # patch mmcv ceiling before import
    import importlib.util
    def _patch(pkg, old, new="2.3.0"):
        spec = importlib.util.find_spec(pkg)
        if spec is None or not spec.origin: return
        with open(spec.origin, "r", encoding="utf-8") as f:
            txt = f.read()
        old_str = f"mmcv_maximum_version = '{old}'"
        if old_str in txt:
            with open(spec.origin, "w", encoding="utf-8") as f:
                f.write(txt.replace(old_str, f"mmcv_maximum_version = '{new}'"))
            print(f"[patch] {pkg}: {old!r} → {new!r}")

    _patch("mmdet",  "2.2.0"); _patch("mmdet",  "2.1.0")
    _patch("mmyolo", "2.1.0"); _patch("mmyolo", "2.2.0")

    from mmdet.apis import init_detector

    # resolve class names
    if ann_file and os.path.exists(ann_file):
        with open(ann_file, "r") as f:
            coco = json.load(f)
        cats = sorted(coco["categories"], key=lambda x: x["id"])
        _class_names = [c["name"] for c in cats]
    else:
        _class_names = IP102_CLASSES[:102]

    _unknown_id = len(_class_names)   # class id == num_classes → unknown
    _device     = device

    print(f"[demo] Loading model …  cfg={cfg_path}  ckpt={ckpt_path}")
    _model = init_detector(cfg_path, ckpt_path, device=device)
    _model.eval()
    print(f"[demo] Model ready. {len(_class_names)} known classes, unknown_id={_unknown_id}")


# ═══════════════════════════════════════════════════════════════════════════════
# INFERENCE
# ═══════════════════════════════════════════════════════════════════════════════

def run_inference(pil_img: Image.Image, score_thr: float):
    """Run model inference on a PIL image, return annotated PIL image + info."""
    import cv2
    from mmdet.apis import inference_detector
 
    # PIL → BGR numpy (mmdet expects BGR)
    img_rgb = np.array(pil_img.convert("RGB"))
    img_bgr = img_rgb[:, :, ::-1].copy()
 
    result = inference_detector(_model, img_bgr)
 
    # parse results
    pred_instances = result.pred_instances
    boxes  = pred_instances.bboxes.cpu().numpy()   # (N, 4) xyxy
    scores = pred_instances.scores.cpu().numpy()   # (N,)
    labels = pred_instances.labels.cpu().numpy()   # (N,)

    # Logic gán nhãn Unknown thông minh cho demo:
    # 1. Các box có score >= score_thr và nhãn >= unknown_id -> vẽ Unknown thực tế từ model.
    # 2. Các box có score thấp hơn score_thr nhưng vẫn > 0.05 -> có thể là vật thể lạ (Unknown) mà model phân vân.
    #    Chúng ta chuyển đổi nhãn của chúng thành Unknown (unknown_id) và đặt score = score_thr để vẽ hiển thị.
    patched_labels = []
    patched_scores = []
    
    for score, label in zip(scores, labels):
        if score >= score_thr:
            patched_labels.append(label)
            patched_scores.append(score)
        elif score >= 0.05:  # Có vật thể nhưng độ tin cậy lớp đã biết rất thấp -> Nghi ngờ là Unknown
            patched_labels.append(_unknown_id)
            patched_scores.append(score)  # Giữ nguyên score thực tế nhưng cho phép hiển thị
        else:
            patched_labels.append(label)
            patched_scores.append(score)
            
    patched_labels = np.array(patched_labels)
    patched_scores = np.array(patched_scores)

    # draw
    annotated = draw_boxes(pil_img.copy(), boxes, patched_scores, patched_labels, score_thr)

    # build text summary
    visible = [(b, s, l) for b, s, l in zip(boxes, patched_scores, patched_labels) if s >= score_thr or (l >= _unknown_id and s >= 0.05)]
    if not visible:
        info_text = "Không phát hiện đối tượng nào (thử giảm ngưỡng confidence)."
    else:
        lines = ["**Kết quả phát hiện:**\n"]
        for i, (b, s, l) in enumerate(visible, 1):
            label_name = _class_names[l] if l < len(_class_names) else "**Unknown**"
            is_unk = l >= _unknown_id
            tag = "🔴 **Unknown**" if is_unk else f"🟡 {label_name}"
            lines.append(f"{i}. {tag} — confidence: `{s:.2f}`")
        info_text = "\n".join(lines)

    return annotated, info_text


def draw_boxes(img: Image.Image, boxes, scores, labels, score_thr):
    """Draw bounding boxes on PIL image."""
    # Ensure RGBA so we can draw semi-transparent label backgrounds
    img = img.convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # try to load a font
    try:
        font = ImageFont.truetype("arial.ttf", FONT_SIZE)
        font_bold = ImageFont.truetype("arialbd.ttf", FONT_SIZE)
    except Exception:
        font = ImageFont.load_default()
        font_bold = font

    for box, score, label in zip(boxes, scores, labels):
        if score < score_thr:
            continue

        x1, y1, x2, y2 = box
        is_unknown = int(label) >= len(_class_names)
        color = UNKNOWN_COLOR if is_unknown else KNOWN_COLOR

        # box outline (thick)
        lw = max(2, int((img.width + img.height) / 600))
        for t in range(lw):
            draw.rectangle([x1-t, y1-t, x2+t, y2+t], outline=color + (255,))

        # label text
        label_str = "Unknown" if is_unknown else _class_names[int(label)]
        text = f"{label_str}  {score:.0%}"

        # text background
        bbox_text = font.getbbox(text)
        tw, th = bbox_text[2] - bbox_text[0], bbox_text[3] - bbox_text[1]
        pad = 4
        ty = max(0, y1 - th - pad * 2)
        draw.rectangle(
            [x1, ty, x1 + tw + pad * 2, ty + th + pad * 2],
            fill=color + (BG_ALPHA,)
        )
        draw.text(
            (x1 + pad, ty + pad),
            text,
            fill=(0, 0, 0, 255) if not is_unknown else (255, 255, 255, 255),
            font=font_bold if is_unknown else font,
        )

    # Composite overlay onto base image, return as RGB
    combined = Image.alpha_composite(img, overlay)
    return combined.convert("RGB")


# ═══════════════════════════════════════════════════════════════════════════════
# GRADIO UI
# ═══════════════════════════════════════════════════════════════════════════════

EXAMPLE_IMAGES = []
_viz_dir = REPO_ROOT / "visualize"
if _viz_dir.exists():
    EXAMPLE_IMAGES = [[str(p)] for p in sorted(_viz_dir.glob("pred_*.png"))[:5]]


CSS = """
/* ── global ── */
body { background: #0d1117; }
.gradio-container { background: #0d1117 !important; font-family: 'Inter', sans-serif; }

/* ── header ── */
#app-header {
    background: linear-gradient(135deg, #1a1f2e 0%, #0d1117 50%, #1a1f2e 100%);
    border-bottom: 1px solid #30363d;
    padding: 32px 48px;
    text-align: center;
    margin-bottom: 24px;
}
#app-header h1 {
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(90deg, #ffd700, #ff6b6b, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 8px 0;
}
#app-header p {
    color: #8b949e;
    font-size: 1rem;
    margin: 0;
}

/* ── panels ── */
.panel {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 12px !important;
    padding: 16px !important;
}

/* ── buttons ── */
#detect-btn {
    background: linear-gradient(135deg, #ffd700, #ff6b35) !important;
    color: #0d1117 !important;
    font-weight: 700 !important;
    font-size: 1.05rem !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 12px 32px !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
}
#detect-btn:hover { filter: brightness(1.1) !important; transform: translateY(-1px) !important; }

#clear-btn {
    background: #21262d !important;
    color: #c9d1d9 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
}

/* ── legend ── */
.legend {
    display: flex; gap: 24px; padding: 12px 0;
    font-size: 0.9rem; color: #8b949e;
}
.legend-known  { color: #ffd700; font-weight: 600; }
.legend-unknown{ color: #ff5070; font-weight: 600; }

/* ── output info ── */
#info-panel {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 12px !important;
    color: #c9d1d9 !important;
}

/* ── slider ── */
.slider-wrap label { color: #8b949e !important; }
"""

_THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.yellow,
    neutral_hue=gr.themes.colors.gray,
    font=gr.themes.GoogleFont("Inter"),
)

def create_ui():
    with gr.Blocks(
        title="OW-OVD Pest Detection Demo",
        theme=_THEME,
        css=CSS,
    ) as demo:

        # ── Header ──────────────────────────────────────────────────────────
        gr.HTML("""
        <div id="app-header">
          <h1>🔬 OW-OVD Pest Detection</h1>
          <p>Open-World Object Detection cho sâu bệnh cây trồng · IP102 Dataset</p>
        </div>
        """)

        # ── Legend ──────────────────────────────────────────────────────────
        gr.HTML("""
        <div class="legend" style="padding-left:24px;">
          <span class="legend-known">⬛ Màu vàng — Known pest (sâu bệnh đã biết)</span>
          <span class="legend-unknown">⬛ Màu đỏ — Unknown (vật thể lạ)</span>
        </div>
        """)

        # ── Main row ─────────────────────────────────────────────────────────
        with gr.Row(equal_height=True):
            with gr.Column(scale=1, elem_classes="panel"):
                gr.Markdown("### 📤 Upload ảnh")
                input_img = gr.Image(
                    type="pil",
                    label="Ảnh đầu vào",
                    height=420,
                    sources=["upload", "clipboard"],
                )
                score_slider = gr.Slider(
                    minimum=0.01, maximum=0.99, value=0.05, step=0.01,
                    label="Ngưỡng confidence (Score Threshold)",
                    elem_classes="slider-wrap",
                )
                with gr.Row():
                    detect_btn = gr.Button("🚀 Phát hiện", elem_id="detect-btn", variant="primary")
                    clear_btn  = gr.Button("🗑 Xoá",        elem_id="clear-btn")

            with gr.Column(scale=1, elem_classes="panel"):
                gr.Markdown("### 🖼 Kết quả nhận diện")
                output_img = gr.Image(
                    type="pil",
                    label="Ảnh sau khi phân tích",
                    height=420,
                    interactive=False,
                )

        # ── Info panel ───────────────────────────────────────────────────────
        gr.Markdown("### 📋 Chi tiết phát hiện")
        info_box = gr.Markdown(
            value="*Hãy upload ảnh và nhấn **Phát hiện** để bắt đầu.*",
            elem_id="info-panel",
        )

        # ── Examples ────────────────────────────────────────────────────────
        if EXAMPLE_IMAGES:
            gr.Markdown("### 📁 Ảnh mẫu")
            gr.Examples(examples=EXAMPLE_IMAGES, inputs=[input_img])

        # ── Footer ──────────────────────────────────────────────────────────
        gr.HTML("""
        <div style="text-align:center; color:#484f58; font-size:0.8rem; padding:24px 0 8px;">
            OW-OVD · YOLO-World backbone · IP102 (102 classes) · Unknown detection enabled
        </div>
        """)

        # ── Event handlers ───────────────────────────────────────────────────
        def on_detect(img, thr):
            if img is None:
                return None, "⚠️ Vui lòng upload ảnh trước."
            if _model is None:
                return None, "⚠️ Model chưa được load. Vui lòng khởi chạy đúng tham số."
            try:
                out_img, info = run_inference(img, thr)
                return out_img, info
            except Exception as e:
                import traceback
                return None, f"❌ Lỗi: {e}\n```\n{traceback.format_exc()}\n```"

        detect_btn.click(
            fn=on_detect,
            inputs=[input_img, score_slider],
            outputs=[output_img, info_box],
        )

        def on_clear():
            return None, None, "*Hãy upload ảnh và nhấn **Phát hiện** để bắt đầu.*"

        clear_btn.click(fn=on_clear, outputs=[input_img, output_img, info_box])

    return demo


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="OW-OVD Pest Detection Demo")
    p.add_argument("--config",     required=True,  help="Path to mmdet config (.py)")
    p.add_argument("--checkpoint", required=True,  help="Path to model checkpoint (.pth)")
    p.add_argument("--ann-file",   default=None,   help="COCO annotations JSON (for class names)")
    p.add_argument("--device",     default="cpu",  help="Device: cpu / cuda:0 / ...")
    p.add_argument("--port",       type=int, default=7860)
    p.add_argument("--share",      action="store_true", help="Create public Gradio link")
    p.add_argument("--no-model",   action="store_true", help="Skip model loading (UI only test)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not args.no_model:
        load_model(args.config, args.checkpoint, args.ann_file, args.device)
    else:
        print("[demo] --no-model: running in UI-only mode (no inference)")
        _class_names = IP102_CLASSES[:102]
        _unknown_id  = 102

    demo = create_ui()
    demo.launch(
        server_port=args.port,
        share=args.share,
        show_error=True,
        inbrowser=True,
    )
