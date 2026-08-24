"""
Generates a professional PDF integration guide for frontend developers.
"""

from fpdf import FPDF
from datetime import datetime


class GuidePDF(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(140, 140, 140)
            self.cell(0, 6, "Audio Deepfake Detection API  |  Frontend Integration Guide", align="L")
            self.cell(0, 6, f"Page {self.page_no() - 1}/{{nb}}", align="R", new_x="LMARGIN", new_y="NEXT")
            self.line(10, self.get_y(), 200, self.get_y())
            self.ln(4)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(170, 170, 170)
        self.cell(0, 10, f"Generated {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}", align="C")


def section_heading(pdf, num, title):
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(20, 60, 120)
    pdf.cell(0, 10, f"{num}. {title}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)


def sub_heading(pdf, title):
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(40, 40, 40)
    pdf.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)


def body_text(pdf, text):
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(60, 60, 60)
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
    pdf.set_font("Courier", "", 8)
    lines = code.split("\n")
    for line in lines:
        pdf.set_x(16)
        pdf.cell(184, 4.5, f"  {line}", fill=True, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)


def bullet(pdf, text, bold_prefix=""):
    pdf.set_x(14)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(60, 60, 60)
    w_pre = pdf.get_string_width(bold_prefix)
    pdf.cell(w_pre + 2, 5, bold_prefix)
    pdf.set_font("Helvetica", "", 9)
    pdf.multi_cell(186 - w_pre - 4, 5, text)


def key_value_table(pdf, rows):
    pdf.set_x(12)
    for key, val in rows:
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(50, 50, 50)
        kw = max(pdf.get_string_width(key + ": "), 35)
        pdf.cell(kw + 2, 6, key + ": ")
        pdf.set_font("Courier", "", 9)
        pdf.set_text_color(20, 80, 160)
        pdf.cell(186 - kw - 2, 6, val, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)


def json_block(pdf, json_str):
    pdf.set_fill_color(245, 245, 250)
    pdf.set_text_color(30, 30, 40)
    pdf.set_font("Courier", "", 7.5)
    lines = json_str.split("\n")
    for line in lines:
        pdf.set_x(14)
        pdf.cell(184, 4, f"  {line}", fill=True, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)


def generate():
    pdf = GuidePDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    BASE = "https://audio-deepfake-detector.reddune-ee354d90.francecentral.azurecontainerapps.io"

    # ---------------------------------------------------------------
    # TITLE PAGE
    # ---------------------------------------------------------------
    pdf.ln(20)
    pdf.set_font("Helvetica", "B", 28)
    pdf.set_text_color(20, 60, 120)
    pdf.cell(0, 14, "Audio Deepfake Detection API", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 16)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 10, "Frontend Integration Guide", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(8)

    pdf.set_draw_color(20, 60, 120)
    pdf.line(50, pdf.get_y(), 160, pdf.get_y())
    pdf.ln(8)

    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 6, "Version 1.0  |  For frontend engineers integrating the deepfake detection service", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, "Classification: Internal  |  Audience: Frontend Development Team", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10)

    pdf.set_draw_color(180, 180, 180)
    pdf.rect(25, pdf.get_y(), 160, 40)
    pdf.set_xy(28, pdf.get_y() + 4)
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(60, 60, 60)
    pdf.cell(154, 6, "Quick Start")
    pdf.set_x(28)
    pdf.set_font("Courier", "", 9)
    pdf.set_text_color(20, 80, 160)
    pdf.cell(154, 6, f"const API = '{BASE}/api/v1';")
    pdf.set_x(28)
    pdf.set_font("Courier", "", 9)
    pdf.set_text_color(20, 80, 160)
    pdf.cell(154, 6, 'const form = new FormData(); form.append("file", audioFile);')
    pdf.set_x(28)
    pdf.set_font("Courier", "", 9)
    pdf.set_text_color(20, 80, 160)
    pdf.cell(154, 6, "fetch(`${API}/detect`, { method: 'POST', body: form })")

    # ---------------------------------------------------------------
    # 1. OVERVIEW
    # ---------------------------------------------------------------
    pdf.add_page()
    section_heading(pdf, "1", "Overview")
    body_text(pdf,
        "The Audio Deepfake Detection API is a production-grade REST microservice that analyzes audio files "
        "and determines whether they contain authentic human speech (bona-fide) or artificially generated "
        "deepfake / voice clone audio (spoof). It is designed for easy integration into any frontend "
        "application  --  dashboard, content moderation UI, media player, or browser extension."
    )

    sub_heading(pdf, "Base URL")
    code_block(pdf, BASE)

    sub_heading(pdf, "Supported Audio Formats")
    bullet(pdf, "WAV, FLAC, MP3, M4A, OGG")
    bullet(pdf, "Maximum file size: 50 MB")
    bullet(pdf, "Maximum duration: 300 seconds (5 minutes)")
    bullet(pdf, "Sample rate: automatically handled (internally resampled to 16 kHz)")

    sub_heading(pdf, "Architecture")
    body_text(pdf,
        "This service uses a three-model ensemble: Wav2Vec2 (HuggingFace transformer for semantic analysis), "
        "AASIST (graph-based anti-spoofing for acoustic artifact detection), and a custom dual-branch "
        "meta-classifier MLP that fuses their outputs into a single confidence score. Every request also "
        "generates XAI attention maps (temporal saliency timeline + heatmap image) and a downloadable "
        "PDF forensic report."
    )

    # ---------------------------------------------------------------
    # 2. ENDPOINT REFERENCE
    # ---------------------------------------------------------------
    pdf.add_page()
    section_heading(pdf, "2", "Endpoint Reference")

    # 2.1 Health
    sub_heading(pdf, "2.1  Health Check  |  GET /api/v1/health")
    body_text(pdf,
        "Use this endpoint to verify the service is running and all models are loaded before sending audio. "
        "Returns load status for Wav2Vec2, AASIST, and the Meta-Classifier."
    )
    key_value_table(pdf, [
        ("Method", "GET"),
        ("URL", f"{BASE}/health"),
        ("Content-Type", "application/json"),
    ])
    sub_heading(pdf, "Response")
    json_block(pdf, """{
  "status": "ok",
  "wav2vec2_loaded": true,
  "aasist_loaded": true,
  "meta_classifier_loaded": true,
  "device": "cpu",
  "uptime_seconds": 1234.5
}""")
    bullet(pdf, "All three *_loaded fields must be true for detection to work")
    bullet(pdf, "If any model is false, notify operations before sending audio")

    # 2.2 Detect
    pdf.add_page()
    sub_heading(pdf, "2.2  Deepfake Detection  |  POST /api/v1/detect")
    body_text(pdf,
        "Upload an audio file for deepfake analysis. This is the primary endpoint. "
        "The request must be multipart/form-data with the file in the 'file' field. "
        "The response includes the verdict, confidence score, per-model breakdown, XAI attention maps, "
        "and a link to download the PDF report."
    )
    key_value_table(pdf, [
        ("Method", "POST"),
        ("URL", f"{BASE}/detect"),
        ("Content-Type", "multipart/form-data"),
        ("Body Field", 'file (UploadFile, required)'),
    ])

    sub_heading(pdf, "Request Example (cURL)")
    code_block(pdf, f'''curl -X POST \\
  -F "file=@sample_audio.wav" \\
  {BASE}/detect''')

    sub_heading(pdf, "Full Response JSON")
    json_block(pdf, f"""{{
  "request_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "prediction": "spoof",
  "confidence": 0.9498,
  "scores": {{
    "wav2vec2": 0.0803,
    "aasist": 0.9907,
    "ensemble": 0.9498
  }},
  "process_time_ms": 1477.9,
  "attention_maps": {{
    "overall_heatmap": "iVBORw0KGgoAAAANS...",
    "layer_count": 12,
    "timeline": [
      {{"start_ms": 0, "end_ms": 160,
        "attention_score": 0.267}},
      {{"start_ms": 160, "end_ms": 320,
        "attention_score": 0.312}}
    ]
  }},
  "report_download_link": "/api/v1/report/
a1b2c3d4-e5f6-7890-abcd-ef1234567890"
}}""")

    sub_heading(pdf, "Response Field Reference")
    pdf.set_x(12)
    rows_data = [
        ("request_id", "string", "UUID for audit trail and report download"),
        ("prediction", "string", '"spoof" (fake) or "bona-fide" (real)'),
        ("confidence", "float", "0.0 - 1.0 ensemble confidence. 0 = real, 1 = spoof"),
        ("scores.wav2vec2", "float", "Wav2Vec2 model confidence (0 - 1)"),
        ("scores.aasist", "float", "AASIST model confidence (0 - 1)"),
        ("scores.ensemble", "float", "Final meta-classifier output (0 - 1)"),
        ("process_time_ms", "float", "Total inference time in milliseconds"),
        ("attention_maps.overall_heatmap", "string", "Base64-encoded PNG heatmap image"),
        ("attention_maps.layer_count", "int", "Number of transformer layers analyzed"),
        ("attention_maps.timeline", "array", "Per-frame attention scores with timestamps"),
        ("report_download_link", "string", "Relative URL to download the PDF report"),
    ]
    for field, ftype, desc in rows_data:
        pdf.set_font("Courier", "", 7.5)
        pdf.set_text_color(20, 80, 160)
        pdf.set_x(14)
        pdf.cell(74, 5, field)
        pdf.set_font("Helvetica", "I", 7.5)
        pdf.set_text_color(120, 120, 120)
        pdf.cell(24, 5, ftype)
        pdf.set_font("Helvetica", "", 7.5)
        pdf.set_text_color(60, 60, 60)
        pdf.multi_cell(74, 5, desc)
    pdf.ln(3)

    # 2.3 Report Download
    pdf.add_page()
    sub_heading(pdf, "2.3  Download Report  |  GET /api/v1/report/{request_id}")
    body_text(pdf,
        "Downloads a professional PDF report for the given detection request. The report includes "
        "a SYSTEM VERDICT block, Decision Executive Summary table, Multi-Model Signal Breakdown, "
        "XAI heatmap with temporal saliency analysis, and a Calibration Guide with confidence bar."
    )
    key_value_table(pdf, [
        ("Method", "GET"),
        ("URL", f"{BASE}/report/{{request_id}}"),
        ("Response Type", "application/pdf (binary download)"),
    ])

    sub_heading(pdf, "Response Headers")
    code_block(pdf, '''Content-Type: application/pdf
Content-Disposition: attachment; filename="deepfake_report_a1b2c3d4.pdf"''')

    # ---------------------------------------------------------------
    # 3. FRONTEND INTEGRATION
    # ---------------------------------------------------------------
    pdf.add_page()
    section_heading(pdf, "3", "Frontend Integration  --  JavaScript")

    sub_heading(pdf, "3.1  File Upload with Fetch API")
    body_text(pdf,
        "The simplest way to upload an audio file from a browser or Node.js environment. "
        "Use FormData to attach the file and send as multipart/form-data."
    )
    code_block(pdf, '''const API = "''' + BASE + '''/api/v1";

async function detectDeepfake(audioFile) {
  const formData = new FormData();
  formData.append("file", audioFile);

  const response = await fetch(`${API}/detect`, {
    method: "POST",
    body: formData,
    // Do NOT set Content-Type header manually  -- 
    // browser sets it with boundary automatically
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || "Detection failed");
  }

  return await response.json();
}''', label="JavaScript (Fetch API)")

    sub_heading(pdf, "3.2  File Upload with Axios")
    code_block(pdf, '''import axios from "axios";

const API = "''' + BASE + '''/api/v1";

async function detectDeepfake(audioFile) {
  const formData = new FormData();
  formData.append("file", audioFile);

  const response = await axios.post(`${API}/detect`, formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });

  return response.data;
}''', label="JavaScript (Axios)")

    sub_heading(pdf, "3.3  Complete Integration Example")
    body_text(pdf,
        "A practical example showing file selection, upload, loading state, error handling, "
        "and rendering results."
    )
    code_block(pdf, '''const API = "''' + BASE + '''/api/v1";

const fileInput = document.getElementById("audioFile");
const resultDiv = document.getElementById("result");
const loadingDiv = document.getElementById("loading");

fileInput.addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;

  // Validate file type
  const validExts = ["wav", "flac", "mp3", "m4a", "ogg"];
  const ext = file.name.split(".").pop().toLowerCase();
  if (!validExts.includes(ext)) {
    return alert("Unsupported format: " + ext);
  }

  // Validate file size (50 MB)
  if (file.size > 50 * 1024 * 1024) {
    return alert("File exceeds 50 MB limit");
  }

  loadingDiv.style.display = "block";
  resultDiv.innerHTML = "";

  try {
    const formData = new FormData();
    formData.append("file", file);

    const resp = await fetch(`${API}/detect`, {
      method: "POST",
      body: formData,
    });

    if (!resp.ok) {
      const err = await resp.json();
      throw new Error(err.detail || `HTTP ${resp.status}`);
    }

    const data = await resp.json();
    renderResult(data);
  } catch (err) {
    resultDiv.innerHTML = `<div class="error">
      Error: ${err.message}</div>`;
  } finally {
    loadingDiv.style.display = "none";
  }
});

function renderResult(data) {
  const isSpoof = data.prediction === "spoof";
  const pct = (data.confidence * 100).toFixed(2);

  resultDiv.innerHTML = `
    <div class="verdict ${isSpoof ? 'spoof' : 'bonafide'}">
      ${isSpoof ? "SPOOF (FAKE)" : "BONA-FIDE (REAL)"}
    </div>
    <div class="confidence">
      Confidence: ${pct}%
    </div>
    <div class="scores">
      <div>Wav2Vec2: ${(data.scores.wav2vec2 * 100).toFixed(1)}%</div>
      <div>AASIST: ${(data.scores.aasist * 100).toFixed(1)}%</div>
      <div>Ensemble: ${(data.scores.ensemble * 100).toFixed(1)}%</div>
    </div>
    <div class="meta">
      Request ID: ${data.request_id.slice(0, 8)}...
      Latency: ${data.process_time_ms} ms
    </div>
  `;
}''', label="Full Example")

    sub_heading(pdf, "3.4  Downloading the PDF Report")
    pdf.add_page()
    body_text(pdf,
        "The response includes a report_download_link field. Use it to let users download "
        "or view the PDF report."
    )
    code_block(pdf, '''const API = "''' + BASE + '''/api/v1";

// After receiving detect response (data.report_download_link):

// Option A: Direct download via anchor click
function downloadReport(link) {
  const a = document.createElement("a");
  a.href = API.replace("/api/v1", "") + link;
  a.download = "deepfake_report.pdf";
  a.click();
}

// Option B: Open in new tab
function viewReport(link) {
  window.open(API.replace("/api/v1", "") + link, "_blank");
}

// Option C: Fetch as blob (for display in an iframe)
async function fetchReport(link) {
  const resp = await fetch(API.replace("/api/v1", "") + link);
  const blob = await resp.blob();
  const url = URL.createObjectURL(blob);
  document.getElementById("pdfViewer").src = url;
}''', label="PDF Download Options")

    sub_heading(pdf, "3.5  Rendering the XAI Heatmap")
    body_text(pdf,
        "The overall_heatmap field is a base64-encoded PNG image. Decode it and display "
        "as an inline image in the browser."
    )
    code_block(pdf, '''function renderHeatmap(base64png) {
  const img = document.createElement("img");
  img.src = "data:image/png;base64," + base64png;
  img.alt = "XAI Attention Heatmap";
  img.style.width = "100%";
  img.style.maxWidth = "600px";
  document.getElementById("heatmap").appendChild(img);
}

// Usage:
renderHeatmap(data.attention_maps.overall_heatmap);''', label="Display Heatmap")

    sub_heading(pdf, "3.6  Attention Timeline Chart")
    body_text(pdf,
        "The timeline array contains per-frame attention scores. You can plot this as a "
        "line chart using any charting library."
    )
    code_block(pdf, '''// Example with Chart.js
function renderTimeline(timeline) {
  const labels = timeline.map(t =>
    (t.start_ms / 1000).toFixed(1) + "s"
  );
  const scores = timeline.map(t => t.attention_score);

  new Chart(document.getElementById("chart"), {
    type: "line",
    data: {
      labels,
      datasets: [{
        label: "Attention Score",
        data: scores,
        borderColor: "crimson",
        backgroundColor: "rgba(220, 20, 60, 0.1)",
        fill: true,
        tension: 0.4,
      }],
    },
    options: {
      responsive: true,
      scales: {
        y: { min: 0, max: 1, title: { display: true, text: "Score" } },
        x: { title: { display: true, text: "Time (s)" } },
      },
      plugins: {
        title: { display: true,
          text: "Wav2Vec2 Attention - Temporal Saliency" },
      },
    },
  });
}''', label="Chart.js Timeline")

    # ---------------------------------------------------------------
    # 4. INTERPRETING RESULTS
    # ---------------------------------------------------------------
    pdf.add_page()
    section_heading(pdf, "4", "Interpreting Results")

    sub_heading(pdf, "4.1  Verdict Classification")
    body_text(pdf,
        "The prediction field provides the final verdict. Use the confidence score to "
        "gauge reliability. The ensemble score is the final output."
    )

    pdf.set_x(12)
    rows = [
        ("0.00 - 0.45", "bona-fide", "Authentic human voice  --  accept"),
        ("0.45 - 0.55", "uncertain", "Decision boundary  --  flag for manual review"),
        ("0.55 - 1.00", "spoof", "Deepfake detected  --  reject or escalate"),
    ]
    for rng, pred, note in rows:
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(50, 50, 50)
        pdf.cell(28, 6, rng, border=1, align="C")
        pdf.cell(28, 6, pred, border=1, align="C")
        pdf.set_font("Helvetica", "", 9)
        pdf.cell(130, 6, note, border=1, align="L", new_x="LMARGIN", new_y="NEXT")

    pdf.ln(4)

    sub_heading(pdf, "4.2  Per-Model Score Breakdown")
    body_text(pdf,
        "Each model in the ensemble captures different spoofing signals:"
    )
    bullet(pdf, "Sensitive to acoustic artifacts (synthetic tones, vocoder fingerprints, re-encoding noise)", bold_prefix="AASIST: ")
    bullet(pdf, "Analyzes semantic token distribution and linguistic coherence. Less sensitive to clean TTS but catches lexical anomalies.", bold_prefix="Wav2Vec2: ")
    bullet(pdf, "Learns when to trust each sub-model based on feature patterns. The final decision authority.", bold_prefix="Ensemble: ")

    sub_heading(pdf, "4.3  XAI Attention Maps")
    body_text(pdf,
        "The timeline array provides per-frame attention scores (0 - 1). High attention regions "
        "indicate where the Wav2Vec2 model focused during inference. Peaks at unexpected time "
        "locations (e.g., mid-sentence spikes) often correlate with synthetic artifacts. "
        "The heatmap image visualizes this as a color-coded temporal saliency chart."
    )

    # ---------------------------------------------------------------
    # 5. ERROR HANDLING
    # ---------------------------------------------------------------
    pdf.add_page()
    section_heading(pdf, "5", "Error Handling")

    body_text(pdf,
        "The API returns standard HTTP status codes. Always check the response status before "
        "parsing the body."
    )

    pdf.set_x(12)
    errors = [
        ("400", "Bad Request", "Unsupported file format. Use .wav, .flac, .mp3, .m4a, .ogg"),
        ("400", "No File", "The 'file' field is missing from the multipart upload"),
        ("413", "File Too Large", "File exceeds the 50 MB limit. Compress or trim before upload"),
        ("422", "Validation Error", "Invalid request format. Check multipart structure"),
        ("500", "Internal Error", "Model inference failed. Retry; if persistent, check /health"),
        ("404", "Report Not Found", "Report ID has expired or never existed. Regenerate via /detect"),
    ]
    for code, title, desc in errors:
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(50, 50, 50)
        pdf.cell(18, 6, code, border=1, align="C")
        pdf.cell(40, 6, title, border=1, align="L")
        pdf.set_font("Helvetica", "", 8.5)
        pdf.cell(128, 6, desc, border=1, align="L", new_x="LMARGIN", new_y="NEXT")

    pdf.ln(4)

    sub_heading(pdf, "5.1  Frontend Error Handling Pattern")
    code_block(pdf, '''async function safeDetect(file) {
  try {
    const resp = await fetch(`${API}/detect`, {
      method: "POST",
      body: buildForm(file),
    });

    // Handle non-2xx responses
    if (resp.status === 413) {
      return { error: "File too large (max 50 MB)" };
    }
    if (resp.status === 400) {
      const err = await resp.json();
      return { error: err.detail || "Invalid file" };
    }
    if (!resp.ok) {
      return { error: `Server error (${resp.status})` };
    }

    return await resp.json();
  } catch (err) {
    // Network error (offline, DNS, timeout)
    return { error: "Network error  --  check your connection" };
  }
}''', label="Robust Error Handling")

    # ---------------------------------------------------------------
    # 6. RATE LIMITS & BEST PRACTICES
    # ---------------------------------------------------------------
    pdf.add_page()
    section_heading(pdf, "6", "Best Practices")

    sub_heading(pdf, "6.1  Performance Expectations")
    bullet(pdf, "Typical inference time: 800 - 2000 ms for a 5-second audio on CPU")
    bullet(pdf, "First request may be slower if model is cold-starting")
    bullet(pdf, "The service scales to zero  --  expect ~5s cold start if unused for a while")

    sub_heading(pdf, "6.2  Recommended UI Patterns")
    bullet(pdf, 'Show a loading spinner during detection (especially important for cold starts)', bold_prefix="Loading: ")
    bullet(pdf, 'Display verdict with clear color coding (red = spoof, green = bona-fide, yellow = borderline 0.45-0.55)', bold_prefix="Color: ")
    bullet(pdf, "Store request_id locally so the PDF report link remains accessible", bold_prefix="Caching: ")
    bullet(pdf, "Audio deepfakes are a serious classification  --  let operators review borderline cases", bold_prefix="Manual review: ")

    sub_heading(pdf, "6.3  Integration Checklist")
    pdf.set_x(14)
    items = [
        " Before each session, call GET /health to verify all models are loaded",
        " Validate file extension and size on the client before uploading",
        " Handle 413 (too large) and 400 (unsupported format) gracefully",
        " Display loading state while waiting for inference (may take several seconds)",
        " Render heatmap image from base64 for visual explainability",
        " Provide a download link/button for the PDF report",
        " Log request_ids for audit trail and debugging",
        " In spoof cases, include a manual override option for operators",
    ]
    for i, item in enumerate(items, 1):
        pdf.set_x(14)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(60, 60, 60)
        pdf.cell(8, 6, f"{i}.")
        pdf.multi_cell(170, 6, item)
    pdf.ln(3)

    # ---------------------------------------------------------------
    # 7. QUICK REFERENCE
    # ---------------------------------------------------------------
    pdf.add_page()
    section_heading(pdf, "7", "Quick Reference Card")

    pdf.set_draw_color(20, 60, 120)
    pdf.set_fill_color(240, 245, 255)
    pdf.rect(12, pdf.get_y(), 186, 55)
    y0 = pdf.get_y()
    pdf.set_xy(15, y0 + 3)
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(20, 60, 120)
    pdf.cell(0, 6, "API ENDPOINTS")
    pdf.set_xy(15, y0 + 11)
    pdf.set_font("Courier", "", 8.5)
    pdf.set_text_color(30, 30, 30)
    lines = [
        f"GET  {BASE}/health",
        f"POST {BASE}/detect  (multipart/form-data, field: 'file')",
        f"GET  {BASE}/report/{{request_id}}  (returns PDF)",
    ]
    for line in lines:
        pdf.set_x(15)
        pdf.cell(180, 5, line, new_x="LMARGIN", new_y="NEXT")

    pdf.set_y(y0 + 30)
    pdf.set_xy(15, pdf.get_y())
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(20, 60, 120)
    pdf.cell(0, 6, "KEY CONSTRAINTS")
    pdf.set_xy(15, pdf.get_y())
    pdf.set_font("Courier", "", 8.5)
    pdf.set_text_color(30, 30, 30)
    pdf.multi_cell(180, 5,
        "Supported: .wav .flac .mp3 .m4a .ogg\n"
        "Max size: 50 MB\n"
        "Max duration: 300 seconds\n"
        "Typical latency: 800-2000 ms"
    )

    pdf.ln(6)

    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 6, "For questions or support, contact the backend team.", align="C", new_x="LMARGIN", new_y="NEXT")

    # ---------------------------------------------------------------
    # Output
    # ---------------------------------------------------------------
    path = "frontend_integration_guide.pdf"
    pdf.output(path)
    return path


if __name__ == "__main__":
    path = generate()
    print(f"Guide generated: {path}")
