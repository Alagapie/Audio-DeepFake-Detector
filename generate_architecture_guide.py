"""
Generates a comprehensive System Architecture PDF for the Audio Deepfake Detector.
"""

from fpdf import FPDF
from datetime import datetime


class ArchPDF(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(140, 140, 140)
            self.cell(0, 6, "Audio Deepfake Detector  |  System Architecture", align="L")
            self.cell(0, 6, f"Page {self.page_no() - 1}/{{nb}}", align="R", new_x="LMARGIN", new_y="NEXT")
            self.line(10, self.get_y(), 200, self.get_y())
            self.ln(4)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(170, 170, 170)
        self.cell(0, 10, f"Generated {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}", align="C")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def section(pdf, num, title):
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(20, 60, 120)
    pdf.cell(0, 10, f"{num}. {title}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)


def sub(pdf, title):
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(40, 40, 40)
    pdf.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)


def body(pdf, text):
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(60, 60, 60)
    pdf.set_x(12)
    pdf.multi_cell(186, 5, text)
    pdf.ln(1)


def mono(pdf, text):
    pdf.set_font("Courier", "", 9)
    pdf.set_text_color(20, 80, 160)
    pdf.set_x(12)
    pdf.multi_cell(186, 5, text)
    pdf.ln(1)


def code_block(pdf, code, label=""):
    if label:
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_text_color(80, 80, 80)
        pdf.set_x(14)
        pdf.cell(0, 5, label, new_x="LMARGIN", new_y="NEXT")
    pdf.set_fill_color(240, 240, 245)
    pdf.set_text_color(30, 30, 40)
    pdf.set_font("Courier", "", 7.5)
    for line in code.split("\n"):
        pdf.set_x(16)
        pdf.cell(184, 4.2, f"  {line}", fill=True, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)


def bullet(pdf, text, bold_prefix=""):
    pdf.set_x(14)
    if bold_prefix:
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(60, 60, 60)
        w = pdf.get_string_width(bold_prefix)
        pdf.cell(w + 2, 5, bold_prefix)
        pdf.set_font("Helvetica", "", 9)
    else:
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(60, 60, 60)
    pdf.multi_cell(170, 5, text)


def ascii_box(pdf, lines, label=""):
    if label:
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_text_color(80, 80, 80)
        pdf.set_x(14)
        pdf.cell(0, 5, label, new_x="LMARGIN", new_y="NEXT")
    pdf.set_fill_color(245, 245, 250)
    pdf.set_text_color(30, 30, 40)
    pdf.set_font("Courier", "", 7)
    for line in lines:
        pdf.set_x(16)
        pdf.cell(184, 4, f"  {line}", fill=True, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)


def key_val(pdf, k, v):
    pdf.set_x(12)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(50, 50, 50)
    kw = max(pdf.get_string_width(k + ": "), 35)
    pdf.cell(kw + 2, 6, k + ": ")
    pdf.set_font("Courier", "", 9)
    pdf.set_text_color(20, 80, 160)
    pdf.cell(186 - kw - 2, 6, v, new_x="LMARGIN", new_y="NEXT")


def draw_arrow(pdf, x1, y1, x2, y2, color=(100, 100, 100)):
    pdf.set_draw_color(*color)
    pdf.set_line_width(0.4)
    pdf.line(x1, y1, x2, y2)
    # arrowhead
    dx, dy = x2 - x1, y2 - y1
    angle = 0.4
    s = 3
    import math
    a = math.atan2(dy, dx)
    pdf.line(x2, y2, x2 - s * math.cos(a - angle), y2 - s * math.sin(a - angle))
    pdf.line(x2, y2, x2 - s * math.cos(a + angle), y2 - s * math.sin(a + angle))


def flow_box(pdf, x, y, w, h, text, fill=(220, 230, 245), text_color=(20, 40, 80), font_style="B", font_size=8):
    pdf.set_fill_color(*fill)
    pdf.set_text_color(*text_color)
    pdf.set_font("Helvetica", font_style, font_size)
    pdf.rect(x, y, w, h, "DF")
    # center text
    tw = pdf.get_string_width(text)
    pdf.set_xy(x + (w - tw) / 2, y + (h - 4) / 2)
    pdf.cell(tw, 4, text)


# ---------------------------------------------------------------------------
# Generate
# ---------------------------------------------------------------------------
def generate():
    pdf = ArchPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # ===== TITLE PAGE =====
    pdf.ln(20)
    pdf.set_font("Helvetica", "B", 28)
    pdf.set_text_color(20, 60, 120)
    pdf.cell(0, 14, "Audio Deepfake Detection API", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 16)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 10, "System Architecture Document", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(8)
    pdf.set_draw_color(20, 60, 120)
    pdf.line(50, pdf.get_y(), 160, pdf.get_y())
    pdf.ln(8)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 6, "Version 1.0  |  Full Technical Deep-Dive", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, "Audience: Engineering, MLOps & Security Teams", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10)

    pdf.set_draw_color(180, 180, 180)
    pdf.rect(25, pdf.get_y(), 160, 30)
    pdf.set_xy(28, pdf.get_y() + 4)
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(60, 60, 60)
    pdf.cell(154, 6, "Repository")
    pdf.set_x(28)
    pdf.set_font("Courier", "", 9)
    pdf.set_text_color(20, 80, 160)
    pdf.cell(154, 6, "C:\\Users\\DELL\\Desktop\\Audio")
    pdf.set_x(28)
    pdf.set_font("Courier", "", 9)
    pdf.set_text_color(20, 80, 160)
    pdf.cell(154, 6, "app/  weights/  docker/  research_and_training/")

    # ===== 1. SYSTEM OVERVIEW =====
    pdf.add_page()
    section(pdf, "1", "System Overview")

    body(pdf,
        "The Audio Deepfake Detection API is a production-grade REST microservice that detects whether an "
        "audio sample contains authentic human speech (bona-fide) or AI-generated deepfake / spoof audio. "
        "It employs a three-model ensemble: Wav2Vec2 (HuggingFace transformer), AASIST (graph-based "
        "anti-spoofing), and a custom dual-branch MLP meta-classifier that fuses embeddings from both "
        "sub-models. Every request automatically generates XAI attention maps and a downloadable PDF "
        "forensic report."
    )

    sub(pdf, "High-Level Architecture Flow")

    ascii_box(pdf, [
        "+-----------+    +----------+    +----------+    +-----------+",
        "|  Client   | -> |  FastAPI | -> |  Audio   | -> | Inference |",
        "| (Browser) |    |  Router  |    |  Preproc |    |Orchestrtr|",
        "+-----------+    +----------+    +----------+    +-----------+",
        "                                              |",
        "                    +-------------------------+",
        "                    v",
        "       +------------+------------+",
        "       |            |            |",
        "       v            v            v",
        "  +---------+  +---------+  +-----------+",
        "  |Wav2Vec2 |  | AASIST  |  |  Meta-    |",
        "  |(768-emb)|  |(160-emb)|  | Classifier|",
        "  +---------+  +---------+  +-----------+",
        "       |            |            |",
        "       +------+-----+            |",
        "              v                  v",
        "       +------------+    +-------------+",
        "       |   XAI     | -> |  PDF Report |",
        "       | Pipeline  |    |  Generator  |",
        "       +------------+    +-------------+",
    ], label="Figure 1: End-to-End Request Flow")

    sub(pdf, "Key Design Decisions")
    bullet(pdf, "CPU-only inference -- no GPU dependency. All three models run on CPU for broader deployment compatibility.")
    bullet(pdf, "Stdin/stdout ffmpeg pipe for non-WAV codecs -- eliminates temp file race conditions and supports MP3, M4A, OGG on-the-fly.")
    bullet(pdf, "Dual-branch Meta-Classifier -- prevents the 768-dim W2V2 embedding from numerically dominating the 160-dim AASIST embedding in a single concatenated layer.")
    bullet(pdf, "Always-on XAI + Report -- every detection call generates attention maps and a PDF report automatically (no query params needed).")
    bullet(pdf, "In-memory report cache -- PDF bytes stored in a dict keyed by UUID request_id (lost on restart).")

    # ===== 2. AUDIO PREPROCESSING =====
    pdf.add_page()
    section(pdf, "2", "Audio Ingestion & Preprocessing")

    sub(pdf, "2.1 File Upload Handling")
    body(pdf,
        "The API receives audio files via multipart/form-data upload (field name: 'file'). "
        "Supported extensions: .wav, .flac, .mp3, .m4a, .ogg. Maximum file size is 50 MB, "
        "maximum duration is 300 seconds (5 minutes)."
    )

    sub(pdf, "2.2 Decoding Pipeline")
    ascii_box(pdf, [
        "Uploaded Bytes",
        "     |",
        "     v",
        "  sf.read()  -- try soundfile first (WAV/FLAC)",
        "     | (fail?)",
        "     v",
        "  ffmpeg stdin pipe: ffmpeg -i pipe:0 -ar 16000 -ac 1 -f wav pipe:1",
        "     |",
        "     v",
        "  + librosa.resample if sample rate != 16000",
        "  + Peak normalization (divide by max abs)",
        "  + Truncate to max_audio_samples (16000 * 300 = 4,800,000)",
        "     |",
        "     v",
        "  np.float32 array, shape (n_samples,)",
    ], label="Figure 2: Audio Ingestion Pipeline")

    code_block(pdf, """# key function: app/utils/audio_utils.py
async def load_audio_from_bytes(data, target_sr=16000):
    try:
        audio, sr = sf.read(io.BytesIO(data))     # WAV/FLAC
    except:
        audio = _convert_with_ffmpeg(data, 16000)  # MP3/M4A/OGG via pipe
    if audio.ndim > 1: audio = audio.mean(axis=1)   # mono
    if sr != 16000: audio = librosa.resample(...)    # resample
    return audio.astype(np.float32), 16000

async def preprocess_pipeline(audio, sr):
    if sr != 16000: audio = librosa.resample(...)
    peak = max(abs(audio))
    if peak > 0: audio /= peak                      # normalize
    if len(audio) > 4800000: audio = audio[:4800000] # truncate
    return audio.astype(np.float32)""", label="Preprocessing Pseudocode")

    sub(pdf, "2.3 ffmpeg Format Support")
    body(pdf,
        "When soundfile (libsndfile) cannot read a file (e.g., MP3, M4A, OGG), the system falls back to "
        "ffmpeg via a stdin-to-stdout pipe. The subprocess reads raw bytes from stdin, converts to 16 kHz "
        "mono WAV, and writes PCM data to stdout. This avoids writing temp files to disk, eliminating race "
        "conditions and cleanup overhead. Availability is checked once at startup via check_ffmpeg()."
    )

    # ===== 3. MODEL: WAV2VEC2 =====
    pdf.add_page()
    section(pdf, "3", "Wav2Vec2 Transformer Pipeline")

    sub(pdf, "3.1 Architecture")
    body(pdf,
        "Wav2Vec2 is a self-supervised speech representation model from Facebook AI / HuggingFace. "
        "This project uses the fine-tuned checkpoint Vansh180/deepfake-audio-wav2vec2 which was "
        "fine-tuned on the ASVspoof 2019 Logical Access dataset for binary anti-spoofing classification "
        "(bona-fide vs spoof)."
    )

    sub(pdf, "3.2 Key Details")
    key_val(pdf, "Model ID", "Vansh180/deepfake-audio-wav2vec2")
    key_val(pdf, "Source", "HuggingFace Hub (auto-downloaded)")
    key_val(pdf, "Hidden dim", "768")
    key_val(pdf, "Attention heads", "12")
    key_val(pdf, "Transformer layers", "12")
    key_val(pdf, "Output head", "Linear(768 -> 2) with Softmax")
    key_val(pdf, "Feature encoder stride", "320 samples (20ms at 16kHz)")
    key_val(pdf, "Attention impl", "eager (required for output_attentions=True)")
    pdf.ln(2)

    body(pdf,
        "The model processes raw audio waveforms. The feature encoder convolves the audio into a "
        "sequence of 768-dim feature vectors at a 20ms frame rate. These vectors pass through 12 "
        "transformer layers with self-attention. The final hidden states are mean-pooled across the "
        "time dimension to produce a single 768-dim utterance-level embedding."
    )

    sub(pdf, "3.3 Inference Modes")

    body(pdf, "The Wav2Vec2Pipeline (app/models/wav2vec2_pipeline.py) exposes three inference methods:")

    bullet(pdf, "Returns (confidence, logits_vector). Uses only the classification head output (no hidden states). Fastest path.", bold_prefix="infer(): ")
    bullet(pdf, "Returns (confidence, mean_pooled_hidden). Also requests hidden_states[-1] from the transformer and mean-pools across time to get a 768-dim embedding. Used for meta-classifier fusion.", bold_prefix="infer_with_embedding(): ")
    bullet(pdf, "Returns (confidence, mean_pooled_hidden, list_of_attention_maps). Enables output_attentions=True, returning a list of 12 attention matrices each of shape (12 heads, seq_len, seq_len). Used for XAI.", bold_prefix="get_attentions(): ")

    sub(pdf, "3.4 Mathematical Formulation")
    body(pdf,
        "Given raw audio x, the feature encoder produces a sequence of frame vectors F = encoder(x), "
        "F in R^{T x 768}. Each transformer layer l applies multi-head self-attention: "
        "A_l = softmax(Q_l K_l^T / sqrt(d_k)), then O_l = A_l V_l. "
        "After 12 layers, we extract H = hidden_states[-1] in R^{T x 768}, then "
        "pooled = mean(H, dim=0) in R^{768}. "
        "The classification head produces logits = W * pooled + b in R^2, "
        "and confidence = softmax(logits)[1] (spoof class probability)."
    )

    # ===== 4. MODEL: AASIST =====
    pdf.add_page()
    section(pdf, "4", "AASIST Anti-Spoofing Model")

    sub(pdf, "4.1 Architecture Overview")
    body(pdf,
        "AASIST (Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks) is a "
        "state-of-the-art model for detecting fake audio. It processes raw waveforms through a Sinc-conv "
        "layer, CNN encoder, and a heterogeneous graph attention mechanism that jointly models spectral "
        "and temporal relationships."
    )

    sub(pdf, "4.2 Model Structure")

    ascii_box(pdf, [
        "Input: raw audio (64600 samples ~ 4 sec @ 16kHz)",
        "  |",
        "  v",
        "  SincConv (70 filters, kernel 128)  -- learnable band-pass",
        "  |",
        "  v",
        "  Abs + MaxPool2d(3,3) + BatchNorm + SELU",
        "  |",
        "  v",
        "  CNN Encoder: 6x ResidualBlock [1->32, 32->32, 32->64, 64->64 x3]",
        "  |",
        "  v",
        "  +-- Spectral branch (max over freq) -> GraphAttentionLayer",
        "  |     +-- Temporal branch (max over time) -> GraphAttentionLayer",
        "  |",
        "  v",
        "  Heterogeneous GAT layers (spectral<->temporal cross-attention)",
        "  |     with dual master nodes + graph pooling",
        "  |",
        "  v",
        "  Concat: T_max, T_avg, S_max, S_avg, master -> Linear(320 -> 2)",
    ], label="Figure 3: AASIST Architecture")

    sub(pdf, "4.3 Layer-by-Layer Breakdown")

    body(pdf, "SincConv: A learnable sinc-based band-pass filter bank applied directly to the raw "
        "waveform. It replaces traditional Mel-filterbanks with parameterized filters that can adapt "
        "during training. The kernel size is 128 samples (8 ms at 16 kHz). 70 filters produce 70 "
        "time-frequency channels.")

    body(pdf, "ResidualBlock: Standard CNN residual blocks with 2D convolutions (kernel 2x3), "
        "SELU activation, batch norm, and max pooling (1x3). The first block maps 1->32 channels, "
        "subsequent blocks progress 32->32, 32->64, 64->64. Six blocks total.")

    body(pdf, "Graph Attention Layers: After the CNN encoder, the feature map is split into "
        "spectral and temporal views. Each view passes through a GraphAttentionLayer that computes "
        "a learned self-attention over nodes (frames for temporal view, frequency bins for spectral "
        "view), then propagates information via attention-weighted aggregation.")

    body(pdf, "Heterogeneous GAT: Four HtrgGraphAttentionLayer modules perform cross-attention "
        "between spectral and temporal branches using two separate master node tracks. This allows "
        "the model to jointly reason about time-frequency relationships. Graph pooling (top-k) "
        "reduces dimensionality at each stage.")

    body(pdf,
        "The final embedding is a concatenation of 5 vectors: T_max, T_avg (max and average "
        "pooled temporal graph), S_max, S_avg (max and average pooled spectral graph), and the "
        "master node. This yields a 160-dim embedding (5 * 32 = 160). The classification head "
        "maps 160 -> 2 for spoof probability.")

    sub(pdf, "4.4 Loading and Weight Details")
    key_val(pdf, "Weights file", "weights/aasist_backbone.pt")
    key_val(pdf, "Source", "Clova AI official AASIST.pth")
    key_val(pdf, "Input length", "64600 samples (~4s)")
    key_val(pdf, "Embedding dim", "160")
    key_val(pdf, "strict=False", "Load ignores missing keys (backbone only)")

    # ===== 5. META-CLASSIFIER =====
    pdf.add_page()
    section(pdf, "5", "Dual-Branch Meta-Classifier")

    sub(pdf, "5.1 Motivation")
    body(pdf,
        "The raw embeddings from Wav2Vec2 (768-dim) and AASIST (160-dim) operate in very different "
        "feature spaces. Simply concatenating them into a single 928-dim vector and feeding it to an MLP "
        "risks having the 768-dim W2V2 embedding numerically dominate the gradient signal. The dual-branch "
        "design processes each embedding through an independent learned projection before fusion."
    )

    sub(pdf, "5.2 Network Architecture")

    ascii_box(pdf, [
        "W2V2 Embedding (768)            AASIST Embedding (160)",
        "      |                               |",
        "      v                               v",
        "  Linear(768 -> 128)              Linear(160 -> 64)",
        "  ReLU                             ReLU",
        "  Dropout(0.3)                     Dropout(0.3)",
        "      |                               |",
        "      +------------+-------------------+",
        "                   |",
        "                   v",
        "              Concat (192)",
        "                   |",
        "                   v",
        "              Linear(192 -> 64)",
        "              ReLU",
        "              Dropout(0.3)",
        "                   |",
        "                   v",
        "              Linear(64 -> 1)",
        "              Sigmoid",
        "                   |",
        "                   v",
        "            P(spoof) in [0, 1]",
    ], label="Figure 4: Dual-Branch MetaMLP Architecture")

    sub(pdf, "5.3 Forward Pass Details")

    code_block(pdf, """class MetaMLP(nn.Module):
    def __init__(self, w2v_dim=768, aasist_dim=160, joint_dim=64):
        self.w2v_branch = nn.Sequential(
            Linear(768, 128), ReLU, Dropout(0.3))
        self.aasist_branch = nn.Sequential(
            Linear(160, 64), ReLU, Dropout(0.3))
        self.classifier = nn.Sequential(
            Linear(192, 64), ReLU, Dropout(0.3),
            Linear(64, 1), Sigmoid())

    def forward(self, x):
        # x.shape: (batch, 928) = cat([w2v_768, aasist_160], dim=-1)
        w2v   = x[:, :768]        # split
        aas   = x[:, 768:]
        w_out = self.w2v_branch(w2v)      # 768->128
        a_out = self.aasist_branch(aas)   # 160->64
        fused = torch.cat([w_out, a_out], dim=-1)  # 192
        return self.classifier(fused)     # 192->64->1->sigmoid""", label="PyTorch Implementation")

    sub(pdf, "5.4 Training Details")
    body(pdf,
        "The MetaMLP was trained on ASVspoof 2019 Logical Access (LA) dataset. Training was performed "
        "in a Kaggle notebook (research_and_training/kaggle_mlp_trainer.py). The training process:"
    )
    bullet(pdf, "Extract W2V2 and AASIST embeddings offline for each training sample")
    bullet(pdf, "Train the dual-branch MLP with BCEWithLogitsLoss")
    bullet(pdf, "Optimizer: AdamW (lr=1e-4, weight_decay=1e-5)")
    bullet(pdf, "Batch size: 64, validation split: 20%")
    bullet(pdf, "Final validation accuracy: 99.17%")
    pdf.ln(1)
    key_val(pdf, "Weights file", "weights/meta_mlp.pt (488 KB)")
    key_val(pdf, "Input dim", "928 (768 + 160)")
    key_val(pdf, "Output", "scalar in [0,1] -- P(spoof)")

    sub(pdf, "5.5 Fallback Strategy")
    body(pdf,
        "If the meta-classifier weights file is missing or fails to load (architecture mismatch), "
        "the system gracefully falls back to simple score averaging: "
        "P_ensemble = (P_w2v + P_aasist) / 2. This is logged as a warning instruction to retrain."
    )

    # ===== 6. ORCHESTRATOR & INFERENCE PIPELINE =====
    pdf.add_page()
    section(pdf, "6", "Inference Orchestrator")

    sub(pdf, "6.1 Orchestration Flow")
    body(pdf,
        "The InferenceOrchestrator (app/inference/orchestrator.py) coordinates the entire detection pipeline. "
        "It is constructed with references to all three models and exposes a single async detect() method."
    )

    ascii_box(pdf, [
        "detect(audio_bytes, xai_enabled=True)",
        "  |",
        "  v",
        "  1. Preprocess audio bytes -> numpy array (16kHz mono)",
        "  |",
        "  v",
        "  2. Wav2Vec2.infer_with_embedding(audio)",
        "     -> (w2v_conf, w2v_emb_768)",
        "  |",
        "  v",
        "  3. AASIST.infer_with_embedding(audio)",
        "     -> (aasist_conf, aasist_emb_160)",
        "  |  (AASIST failure is caught & logged; continues with 0.5 fallback)",
        "  v",
        "  4. MetaClassifier.predict_proba(w2v_emb, aasist_emb, ...)",
        "     -> ensemble_conf",
        "  |",
        "  v",
        "  5. DetectionResult with prediction, confidence, scores, latency",
        "  |",
        "  v",
        "  (if xai_enabled) -> _detect_with_xai()",
        "      |",
        "      +-> Wav2Vec2.get_attentions()  -> 12 raw attention matrices",
        "      +-> process_attentions()        -> Attention Rollout",
        "      +-> map_attention_to_timeline() -> per-frame scores",
        "      +-> generate_heatmap()          -> base64 PNG",
        "      +-> XaiResult(detection, attention_maps)",
    ], label="Figure 5: Orchestrator Sequence Diagram")

    sub(pdf, "6.2 Code Structure")

    code_block(pdf, """class InferenceOrchestrator:
    def __init__(self, wav2vec2, aasist, meta_classifier):
        self._w2v = wav2vec2
        self._aasist = aasist
        self._meta = meta_classifier

    async def detect(self, audio_bytes, xai_enabled=True):
        start = time.perf_counter()
        audio = await preprocess_audio_bytes(audio_bytes)
        w2v_conf, w2v_emb = await self._w2v.infer_with_embedding(audio)
        try:
            aasist_conf, aasist_emb = await self._aasist.infer_with_embedding(audio)
        except Exception:
            aasist_conf, aasist_emb = 0.5, None
        ensemble_conf = await self._meta.predict_proba(w2v_emb, aasist_emb,
                                                       w2v_conf, aasist_conf)
        det = DetectionResult(
            prediction="spoof" if ensemble_conf > 0.5 else "bona-fide",
            confidence=ensemble_conf,
            scores={"wav2vec2": w2v_conf, "aasist": aasist_conf, "ensemble": ensemble_conf},
            process_time_ms=round((time.perf_counter()-start)*1000, 1),
        )
        if xai_enabled:
            return await self._detect_with_xai(audio, det, start)
        return det""", label="Orchestrator Core Logic")

    # ===== 7. XAI PIPELINE =====
    pdf.add_page()
    section(pdf, "7", "Explainable AI (XAI) Pipeline")

    sub(pdf, "7.1 Overview")
    body(pdf,
        "The XAI pipeline provides interpretability for Wav2Vec2's decision by analyzing the "
        "self-attention matrices across all 12 transformer layers. It uses Attention Rollout "
        "to propagate attention through the residual connections, producing a per-frame importance "
        "score that indicates which temporal regions the model focused on during inference."
    )

    sub(pdf, "7.2 Attention Rollout Algorithm")
    body(pdf,
        "Raw attention matrices from individual transformer layers only show direct attention "
        "patterns and ignore the residual connections (skip connections) that carry information "
        "around each self-attention block. Attention Rollout corrects this by mixing each layer's "
        "attention with the identity matrix:"
    )
    body(pdf,
        "For each layer l with attention matrix A_l (averaged over heads):\n"
        "  1. Mix with identity: A'_l = 0.5 * A_l + 0.5 * I\n"
        "  2. Row-normalize: A'_l[i,:] = A'_l[i,:] / sum(A'_l[i,:])\n"
        "  3. Accumulate: rollout = A'_l @ rollout (starting from I)\n"
        "The final rollout matrix captures the total influence flow from input to output across "
        "the entire transformer stack. The per-frame importance is rollout.mean(axis=0) -- the "
        "average amount of attention each input frame receives from all output positions."
    )

    code_block(pdf, """def _attention_rollout(attentions):
    seq_len = attentions[0].shape[-1]
    rollout = np.eye(seq_len)               # start with identity
    for layer_attn in attentions:
        A = layer_attn.mean(axis=0)         # average 12 heads -> (T, T)
        A = 0.5 * A + 0.5 * np.eye(seq_len) # add residual connection
        A = A / A.sum(axis=-1, keepdims=True)# row-normalize
        rollout = A @ rollout               # accumulate
    return rollout                          # (T, T)

def process_attentions(attentions):
    layer_scores = []
    for attn in attentions:                 # per-layer source-aggregated
        layer_scores.append(attn.mean(axis=0).mean(axis=0))
    rollout = _attention_rollout(attentions)
    importance = rollout.mean(axis=0)       # per-frame importance
    layer_scores.append(importance)
    return layer_scores                     # last element is rollout""", label="Attention Rollout Implementation")

    sub(pdf, "7.3 Timeline Mapping")
    body(pdf,
        "The per-frame importance scores from the rollout are mapped to real timestamps using the "
        "feature encoder's stride. Wav2Vec2's feature encoder has a stride of 320 samples (20ms "
        "at 16 kHz). Each frame index i corresponds to time interval "
        "[i * 320 / 16000, (i+1) * 320 / 16000] seconds. The scores are min-max normalized to [0, 1]."
    )

    code_block(pdf, """def map_attention_to_timeline(layer_maps):
    stride = 320  # feature encoder stride in samples
    sr = 16000
    aggregate = layer_maps[-1]  # rollout importance
    aggregate = (aggregate - min) / (max - min + 1e-8)  # normalize
    timeline = []
    for i in range(n_frames):
        timeline.append({
            "frame": i,
            "start_ms": i * stride / sr * 1000,
            "end_ms": (i+1) * stride / sr * 1000,
            "attention_score": float(aggregate[i]),
        })
    return timeline""", label="Temporal Mapping")

    sub(pdf, "7.4 Heatmap Visualization")
    body(pdf,
        "The visualizer (app/xai/visualizer.py) uses matplotlib to render the timeline scores "
        "as a filled line chart (crimson fill, darkred line). The plot is saved to a PNG in-memory "
        "buffer, then base64-encoded for transport in the JSON response and embedding in the PDF report."
    )

    sub(pdf, "7.5 XAI Output Fields")
    bullet(pdf, "Base64-encoded PNG of the attention vs time chart", bold_prefix="overall_heatmap: ")
    bullet(pdf, "Number of transformer layers (always 12 for W2V2 base)", bold_prefix="layer_count: ")
    bullet(pdf, "Array of {frame, start_ms, end_ms, attention_score} objects", bold_prefix="timeline: ")

    # ===== 8. PDF REPORT GENERATOR =====
    pdf.add_page()
    section(pdf, "8", "PDF Report Generator")

    sub(pdf, "8.1 Report Structure")
    body(pdf,
        "The PDF report is generated using fpdf2 (FPDF). Each report contains four major sections:"
    )

    ascii_box(pdf, [
        "+================================================+",
        "|  SYSTEM VERDICT BLOCK                          |",
        "|  (Red=SPOOF or Green=BONA-FIDE full-width bar) |",
        "+================================================+",
        "|  1. Decision Executive Summary                 |",
        "|     Table: Attribute | Value | Interpretation  |",
        "+------------------------------------------------+",
        "|  2. Multi-Model Signal Breakdown               |",
        "|     AASIST: score, focus, result               |",
        "|     Wav2Vec2: score, focus, result             |",
        "|     Ensemble: score, focus, result             |",
        "+------------------------------------------------+",
        "|  3. Visual Explainability (XAI)                |",
        "|     Heatmap image + temporal analysis text     |",
        "+------------------------------------------------+",
        "|  4. Calibration Guide                          |",
        "|     Confidence bar 0.0---[marker]---1.0        |",
        "|     Interpretation policy note                 |",
        "+================================================+",
    ], label="Figure 6: PDF Report Layout")

    sub(pdf, "8.2 Generation Details")
    body(pdf,
        "The report is generated on-the-fly after each detection call. The generate_report() "
        "function in app/utils/pdf_report.py receives the request_id, prediction, confidence, "
        "scores, process_time_ms, and attention_maps dict. It constructs a ReportPDF object, "
        "draws each section using fpdf2 primitives, and returns the PDF as bytes."
    )

    sub(pdf, "8.3 Key Implementation Notes")
    bullet(pdf, "Latin-1 limitation: fpdf2 Helvetica font does not support Unicode. All special characters (em-dashes, smart quotes) are replaced with ASCII equivalents.")
    bullet(pdf, "Table wrapping: A custom _multi_cell_row helper uses multi_cell for text wrapping inside table cells, unlike fixed-height cell() which truncates.")
    bullet(pdf, "Verdict block: Full-width colored rectangle (red #DC3232 or green #28B450) with white text, drawn at the top of the report body.")
    bullet(pdf, "Calibration bar: A dual-colored horizontal bar (green left half, red right half) with a black triangular marker at the confidence position.")
    bullet(pdf, "In-memory cache: PDF bytes stored in the global REPORTS dict keyed by request_id UUID. Retrieved on GET /api/v1/report/{id}.")

    # ===== 9. API LAYER =====
    pdf.add_page()
    section(pdf, "9", "API Layer & FastAPI Application")

    sub(pdf, "9.1 Application Entry Point")

    code_block(pdf, """# app/main.py
app = FastAPI(title="Audio Deepfake Detection API",
              version="1.0.0",
              lifespan=lifespan)
app.add_middleware(CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"])
app.include_router(router)""", label="FastAPI Application Setup")

    sub(pdf, "9.2 Lifespan & Startup")
    body(pdf,
        "The lifespan handler (app/api/lifespan.py) runs on startup: it checks ffmpeg availability, "
        "then initializes and loads all three models asynchronously via init_components(). "
        "On shutdown, it unloads the models to free memory. The startup timestamp is captured for "
        "uptime calculation in the health endpoint."
    )

    sub(pdf, "9.3 Route Map")

    ascii_box(pdf, [
        "GET  /api/v1/health",
        "  Returns: JSON with model load status, device, uptime",
        "",
        "POST /api/v1/detect",
        "  Body: multipart/form-data with 'file' field (UploadFile)",
        "  Returns: JSON with prediction, confidence, scores, XAI maps, report link",
        "",
        "GET  /api/v1/report/{request_id}",
        "  Returns: application/pdf binary download",
    ], label="Figure 7: API Endpoints")

    sub(pdf, "9.4 Request Lifecycle (Detailed)")

    ascii_box(pdf, [
        "1. HTTP Request arrives at FastAPI router",
        "2. CORS middleware checks origin (wildcard: any origin allowed)",
        "3. File extension validated against supported_extensions set",
        "4. File size validated against max_upload_size_mb (50 MB)",
        "5. File bytes read into memory",
        "6. InferenceOrchestrator.detect() called (always xai_enabled=True)",
        "7. UUID request_id generated",
        "8. Response schema populated (DetectResponse or XaiResponse)",
        "9. PDF report generated via generate_report()",
        "10. PDF bytes stored in REPORTS dict with request_id key",
        "11. JSON response returned with report_download_link field",
    ], label="Figure 8: Request Lifecycle")

    sub(pdf, "9.5 Response Schemas")

    code_block(pdf, """class DetectResponse(BaseModel):
    request_id: str
    prediction: str          # "spoof" or "bona-fide"
    confidence: float        # 0.0 to 1.0
    scores: dict[str, float] # wav2vec2, aasist, ensemble
    process_time_ms: float

class XaiAttentionMap(BaseModel):
    overall_heatmap: str     # base64 PNG
    layer_count: int
    timeline: list[dict]     # {start_ms, end_ms, attention_score}

class XaiResponse(DetectResponse):
    attention_maps: XaiAttentionMap | None = None

class HealthResponse(BaseModel):
    status: str              # "ok"
    wav2vec2_loaded: bool
    aasist_loaded: bool
    meta_classifier_loaded: bool
    device: str              # "cpu"
    uptime_seconds: float""", label="Pydantic Response Models")

    # ===== 10. DEPENDENCY INJECTION =====
    pdf.add_page()
    section(pdf, "10", "Dependency Injection & Component Lifecycle")

    sub(pdf, "10.1 AppComponents")
    body(pdf,
        "The AppComponents class (app/dependencies.py) acts as a service container that holds "
        "singleton references to all three model pipelines. It is initialized once at startup "
        "via init_components() and shared across requests through FastAPI's Depends mechanism."
    )

    code_block(pdf, """class AppComponents:
    def __init__(self):
        self.wav2vec2 = Wav2Vec2Pipeline()
        self.aasist = AASISTPipeline()
        self.meta_classifier = MetaClassifier(settings.meta_classifier_path)

    @property
    def orchestrator(self) -> InferenceOrchestrator:
        return InferenceOrchestrator(self.wav2vec2, self.aasist, self.meta_classifier)

async def init_components() -> AppComponents:
    comps = AppComponents()
    await comps.wav2vec2.load()       # downloads from HF Hub
    await comps.aasist.load()         # loads from weights/aasist_backbone.pt
    comps.meta_classifier.load()      # loads from weights/meta_mlp.pt
    return comps

async def get_orchestrator() -> InferenceOrchestrator:
    comps = await init_components()   # returns cached singleton
    return comps.orchestrator""", label="Dependency Container")

    sub(pdf, "10.2 Singleton Pattern")
    body(pdf,
        "The _components module-level variable caches the AppComponents instance after first "
        "initialization. Subsequent calls to init_components() return the same instance, ensuring "
        "models are loaded exactly once. FastAPI routes receive the orchestrator via "
        "Depends(get_orchestrator), which is resolved per-request but returns the shared instance."
    )

    # ===== 11. CONFIGURATION =====
    section(pdf, "11", "Configuration System")

    sub(pdf, "11.1 Settings (Pydantic BaseSettings)")
    body(pdf,
        "All configuration is managed through a Pydantic BaseSettings class (app/config.py). "
        "Settings can be overridden via environment variables with the ADFD_ prefix."
    )

    code_block(pdf, """class Settings(BaseSettings):
    sample_rate: int = 16000
    max_audio_seconds: int = 300
    max_audio_samples: int = sample_rate * max_audio_seconds

    aasist_nb_samp: int = 64600
    device: str = "cpu"
    weights_dir: Path = Path("weights")

    wav2vec2_model_id: str = "Vansh180/deepfake-audio-wav2vec2"
    aasist_pytorch_path: Path = weights_dir / "aasist_backbone.pt"
    aasist_download_url: str = "https://github.com/clovaai/aasist/raw/main/..."
    meta_classifier_path: Path = weights_dir / "meta_mlp.pt"

    xai_layers_to_visualize: int = 4
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 1
    max_upload_size_mb: int = 50
    log_level: str = "INFO"
    model_config = {"env_prefix": "ADFD_"}

    @property
    def supported_extensions(self) -> set[str]:
        return {".wav", ".flac", ".mp3", ".m4a", ".ogg"}""", label="Settings Class")

    # ===== 12. DEPLOYMENT =====
    pdf.add_page()
    section(pdf, "12", "Deployment Architecture")

    sub(pdf, "12.1 Docker Container")

    ascii_box(pdf, [
        "+====================================================+",
        "|  Docker Image: audio-deepfake-detector:latest     |",
        "|  Azure ACR: deepfakeprojectacr2026.azurecr.io     |",
        "+====================================================+",
        "|  Base: python:3.11-slim                           |",
        "|  +-- PyTorch CPU (--index-url cpu)                |",
        "|  +-- ffmpeg (for MP3/M4A/OGG decoding)            |",
        "|  +-- curl (healthcheck)                           |",
        "|  +-- pip install -r requirements.txt              |",
        "|  +-- COPY app/  weights/  docker/                 |",
        "|  EXPOSE 8000                                      |",
        "|  HEALTHCHECK curl -f http://localhost:8000/health  |",
        "|  CMD [\"uvicorn\", \"app.main:app\", \"--host\", \"0.0.0.0\"] |",
        "+====================================================+",
    ], label="Figure 9: Docker Container Structure")

    sub(pdf, "12.2 Azure Deployment")

    ascii_box(pdf, [
        "  Internet",
        "     |",
        "     v",
        "  Azure Container Apps (francecentral)",
        "     |",
        "     +-- Container: audio-deepfake-detector",
        "     |   - CPU-only (no GPU)",
        "     |   - 4 GB memory limit",
        "     |   - Port 8000",
        "     |   - Ingress: external, HTTPS",
        "     |",
        "     +-- ACR: deepfakeprojectacr2026.azurecr.io",
        "     |   - Image: audio-deepfake-detector:latest",
        "     |",
        "     +-- Health probe: GET /api/v1/health",
        "",
        "  URL: https://audio-deepfake-detector.reddune-ee354d90.francecentral.azurecontainerapps.io",
    ], label="Figure 10: Azure Architecture")

    sub(pdf, "12.3 Dockerfile Highlights")
    body(pdf,
        "The Docker build uses a single-stage python:3.11-slim image. The critical optimization is "
        "the --index-url https://download.pytorch.org/whl/cpu flag when installing PyTorch, which "
        "downloads the CPU-only wheel (~120 MB) instead of the CUDA bundle (~2+ GB). This reduces "
        "the build time on Azure ACR from 20+ minutes to approximately 5 minutes and keeps the "
        "final image below 1.5 GB."
    )

    code_block(pdf, """# docker/Dockerfile (key lines)
FROM python:3.11-slim
RUN apt-get update && apt-get install -y ffmpeg curl
COPY requirements.txt .
RUN pip install --no-cache-dir --index-url \\
    https://download.pytorch.org/whl/cpu -r requirements.txt
COPY app/ app/
COPY weights/ weights/
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s \\
    CMD curl -f http://localhost:8000/api/v1/health || exit 1
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]""", label="Dockerfile")

    sub(pdf, "12.4 Docker Compose (Local Dev)")

    code_block(pdf, """# docker/docker-compose.yml
services:
  audio-deepfake-detector:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    ports:
      - "8000:8000"
    volumes:
      - ../weights:/weights
    mem_limit: 4g
    healthcheck:
      test: curl -f http://localhost:8000/api/v1/health || exit 1
      interval: 30s
      timeout: 10s""", label="Docker Compose")

    # ===== 13. DIRECTORY STRUCTURE =====
    pdf.add_page()
    section(pdf, "13", "Project Directory Structure")

    ascii_box(pdf, [
        "C:\\Users\\DELL\\Desktop\\Audio\\",
        "  |",
        "  +-- app/",
        "  |   +-- __init__.py",
        "  |   +-- main.py                 FastAPI app entry",
        "  |   +-- config.py               Pydantic settings",
        "  |   +-- dependencies.py          DI container",
        "  |   |",
        "  |   +-- api/",
        "  |   |   +-- routes.py           3 endpoints (health, detect, report)",
        "  |   |   +-- schemas.py          Pydantic request/response models",
        "  |   |   +-- lifespan.py         Startup/shutdown handler",
        "  |   |",
        "  |   +-- models/",
        "  |   |   +-- wav2vec2_pipeline.py   W2V2 loader + inference",
        "  |   |   +-- aasist_pipeline.py     AASIST model + inference",
        "  |   |   +-- meta_classifier.py     MetaMLP + MetaClassifier",
        "  |   |",
        "  |   +-- inference/",
        "  |   |   +-- orchestrator.py      Coordinates multi-model pipeline",
        "  |   |   +-- preprocessor.py      Audio bytes -> numpy array",
        "  |   |",
        "  |   +-- xai/",
        "  |   |   +-- attention_capturer.py   Attention Rollout algorithm",
        "  |   |   +-- temporal_mapper.py      Frame scores -> timestamps",
        "  |   |   +-- visualizer.py           matplotlib -> base64 PNG",
        "  |   |",
        "  |   +-- utils/",
        "  |       +-- audio_utils.py       ffmpeg pipe, load, normalize",
        "  |       +-- pdf_report.py        fpdf2 report generator",
        "  |",
        "  +-- weights/",
        "  |   +-- aasist_backbone.pt       AASIST pretrained weights",
        "  |   +-- meta_mlp.pt              Dual-branch MLP trained weights",
        "  |",
        "  +-- docker/",
        "  |   +-- Dockerfile               CPU-only PyTorch, ffmpeg",
        "  |   +-- docker-compose.yml       Local dev setup",
        "  |",
        "  +-- research_and_training/",
        "  |   +-- kaggle_mlp_trainer.py    MetaMLP training notebook",
        "  |",
        "  +-- requirements.txt             Python dependencies",
        "  +-- .dockerignore                Excludes venv, notebooks, tests",
        "  +-- .gitignore",
    ], label="Figure 11: File Tree")

    # ===== 14. DATA FLOW SEQUENCE =====
    pdf.add_page()
    section(pdf, "14", "Complete Data Flow Sequence Diagram")

    ascii_box(pdf, [
        "USER                    API                    W2V2                AASIST",
        "  |                      |                      |                   |",
        "  | POST /detect (file)  |                      |                   |",
        "  |--------------------->|                      |                   |",
        "  |                      | preprocess_audio()   |                   |",
        "  |                      |     |                |                   |",
        "  |                      | infer_with_embedding()........................",
        "  |                      |--------------------->|                   |",
        "  |                      |    (conf, 768-emb)   |                   |",
        "  |                      |<---------------------|                   |",
        "  |                      |                      |                   |",
        "  |                      | infer_with_embedding().....................",
        "  |                      |------------------------------------------>|",
        "  |                      |    (conf, 160-emb)   |                   |",
        "  |                      |<------------------------------------------|",
        "  |                      |                      |                   |",
        "  |                      | MetaMLP(w2v_emb + aasist_emb)            |",
        "  |                      |     |                |                   |",
        "  |                      |    ensemble_conf     |                   |",
        "  |                      |                      |                   |",
        "  |                      | get_attentions().....|                   |",
        "  |                      |--------------------->|                   |",
        "  |                      |  12 attention mats   |                   |",
        "  |                      |<---------------------|                   |",
        "  |                      |                      |                   |",
        "  |                      | Attention Rollout    |                   |",
        "  |                      | Timeline Mapping     |                   |",
        "  |                      | Heatmap generation   |                   |",
        "  |                      |                      |                   |",
        "  |                      | generate_report()    |                   |",
        "  |                      |                      |                   |",
        "  |    JSON + report URL |                      |                   |",
        "  |<---------------------|                      |                   |",
        "  |                      |                      |                   |",
        "  | GET /report/{id}     |                      |                   |",
        "  |--------------------->|                      |                   |",
        "  |  PDF (binary)        |                      |                   |",
        "  |<---------------------|                      |                   |",
    ], label="Figure 12: Full Sequence Diagram")

    # ===== 15. TRAINING PIPELINE =====
    pdf.add_page()
    section(pdf, "15", "Meta-Classifier Training Pipeline")

    sub(pdf, "15.1 Training Overview")
    body(pdf,
        "The dual-branch MetaMLP is trained separately from the base models. The process involves "
        "extracting embeddings from W2V2 and AASIST for the ASVspoof 2019 LA dataset, then training "
        "the lightweight MLP (only ~200K parameters) to fuse them optimally."
    )

    sub(pdf, "15.2 Training Steps")
    body(pdf,
        "Step 1 -- Feature Extraction: For each audio file in the ASVspoof 2019 LA training set, "
        "run W2V2.infer_with_embedding() and AASIST.infer_with_embedding() to obtain the 768-dim "
        "and 160-dim embeddings. Save these as a feature cache (numpy arrays) along with labels."
    )
    body(pdf,
        "Step 2 -- Train/Val Split: Split the cached features into 80% training and 20% validation "
        "sets, stratified by label to maintain class balance."
    )
    body(pdf,
        "Step 3 -- MLP Training: Train the MetaMLP with Binary Cross-Entropy loss (BCEWithLogitsLoss). "
        "Use AdamW optimizer with lr=1e-4 and weight_decay=1e-5. Batch size is 64. Train for 50 epochs "
        "with early stopping (patience=5) based on validation accuracy."
    )
    body(pdf,
        "Step 4 -- Evaluation: The final model achieves 99.17% validation accuracy on the held-out "
        "20% of ASVspoof 2019 LA. Weights are saved to weights/meta_mlp.pt (~488 KB)."
    )

    sub(pdf, "15.3 Training code location")
    key_val(pdf, "Script", "research_and_training/kaggle_mlp_trainer.py")

    # ===== 16. PERFORMANCE CHARACTERISTICS =====
    pdf.add_page()
    section(pdf, "16", "Performance & Resource Usage")

    sub(pdf, "16.1 CPU Inference Latency")
    bullet(pdf, "Wav2Vec2 forward pass: ~400-800 ms for 5s audio")
    bullet(pdf, "AASIST forward pass: ~100-200 ms for 4s audio")
    bullet(pdf, "Meta-classifier forward pass: <1 ms")
    bullet(pdf, "XAI attention processing: ~50-100 ms")
    bullet(pdf, "PDF generation: ~200-400 ms")
    bullet(pdf, "Audio preprocessing + ffmpeg decode: ~50-200 ms")
    bullet(pdf, "Total typical: 800-2000 ms (5s audio on CPU)")

    sub(pdf, "16.2 Memory Usage")
    bullet(pdf, "Wav2Vec2 model: ~450 MB (transformer + feature encoder)")
    bullet(pdf, "AASIST model: ~50 MB")
    bullet(pdf, "MetaMLP model: ~2 MB")
    bullet(pdf, "Python runtime + FastAPI: ~100 MB")
    bullet(pdf, "Total (steady state): ~600-700 MB")
    bullet(pdf, "Docker memory limit: 4 GB (provides headroom for requests)")

    sub(pdf, "16.3 Model Sizes")
    bullet(pdf, "Wav2Vec2: ~450 MB (downloaded from HF Hub on first load)")
    bullet(pdf, "AASIST: ~80 MB (weights/aasist_backbone.pt)")
    bullet(pdf, "MetaMLP: ~488 KB (weights/meta_mlp.pt)")

    # ===== 17. DEPENDENCY COMPLETE =====
    pdf.add_page()
    section(pdf, "17", "Dependency Graph")

    ascii_box(pdf, [
        "                          uvicorn",
        "                            |",
        "                          FastAPI",
        "                         /   |   \\",
        "                    pydantic  |  python-multipart",
        "                              |",
        "                    +---------+---------+",
        "                    |         |         |",
        "                    v         v         v",
        "              torch+cpu  transformers  numpy",
        "                    |         |",
        "                    |         +--- soundfile",
        "                    |         +--- librosa",
        "                    |         +--- ffmpeg (system)",
        "                    |",
        "                    +--- matplotlib",
        "                    +--- fpdf2",
        "",
        "  Core: torch, transformers, numpy, soundfile",
        "  Audio: librosa, ffmpeg",
        "  XAI: matplotlib",
        "  Report: fpdf2",
        "  API: FastAPI, uvicorn, pydantic, python-multipart",
    ], label="Figure 13: Dependency Graph")

    # ===== 18. FILE REFERENCE =====
    pdf.add_page()
    section(pdf, "18", "Complete File Reference")

    files = [
        ("app/main.py", "FastAPI app creation, CORS middleware, router inclusion", "22"),
        ("app/config.py", "Pydantic BaseSettings with all configurable parameters", "48"),
        ("app/dependencies.py", "AppComponents container, singleton init, FastAPI Depends factory", "38"),
        ("app/api/routes.py", "3 endpoints: GET /health, POST /detect, GET /report/{id}", "96"),
        ("app/api/schemas.py", "Pydantic models: DetectResponse, XaiResponse, XaiAttentionMap, HealthResponse", "28"),
        ("app/api/lifespan.py", "Startup: ffmpeg check, model loading. Shutdown: model unloading", "24"),
        ("app/models/wav2vec2_pipeline.py", "W2V2: load, preprocess, infer, infer_with_embedding, get_attentions", "86"),
        ("app/models/aasist_pipeline.py", "AASIST: SincConv, GAT layers, AASISTModel, pipeline inference", "320"),
        ("app/models/meta_classifier.py", "MetaMLP (dual-branch), MetaClassifier wrapper with fallback", "99"),
        ("app/inference/orchestrator.py", "InferenceOrchestrator: calls all 3 models, XAI, returns DetectionResult/XaiResult", "79"),
        ("app/inference/preprocessor.py", "Short wrapper: bytes -> numpy via audio_utils + preprocess_pipeline", "10"),
        ("app/xai/attention_capturer.py", "Attention Rollout algorithm, per-layer aggregation", "36"),
        ("app/xai/temporal_mapper.py", "Frame-level attention -> timestamped timeline dicts", "32"),
        ("app/xai/visualizer.py", "Matplotlib: timeline line chart -> base64 PNG heatmap", "61"),
        ("app/utils/audio_utils.py", "ffmpeg pipe, soundfile load, resample, peak normalize, truncate", "68"),
        ("app/utils/pdf_report.py", "fpdf2 report: verdict block, summary table, model cards, XAI, calibration bar", "340"),
        ("docker/Dockerfile", "Multi-stage: python:3.11-slim, CPU-only PyTorch, ffmpeg, curl", "~30"),
        ("docker/docker-compose.yml", "Local dev: port 8000, weights volume, 4GB limit, healthcheck", "~20"),
        ("requirements.txt", "All Python dependencies", "~20"),
        (".dockerignore", "Excludes venv, notebooks, tests, .git", "~10"),
        ("generate_frontend_guide.py", "Generates frontend_integration_guide.pdf for frontend devs", "706"),
        ("generate_architecture_guide.py", "Generates this architecture document", "~1000"),
    ]

    for i, (path, desc, lines) in enumerate(files, 1):
        pdf.set_x(12)
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_text_color(20, 60, 120)
        pdf.cell(8, 5, f"{i}.")
        pdf.cell(72, 5, path)
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(60, 60, 60)
        pdf.cell(86, 5, desc)
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(140, 140, 140)
        pdf.cell(20, 5, f"({lines} lines)", new_x="LMARGIN", new_y="NEXT")

    pdf.ln(6)
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 6, "Total: 22 files | ~2,300 lines of Python | Last updated: June 2026",
             align="C", new_x="LMARGIN", new_y="NEXT")

    # ===== Output =====
    path = "system_architecture_guide.pdf"
    pdf.output(path)
    return path


if __name__ == "__main__":
    path = generate()
    print(f"Architecture guide generated: {path}")
