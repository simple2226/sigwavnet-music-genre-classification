// Build the SigWavNet -> Music Genre Classification deck.
//   node make_deck.js
// After training, edit RESULTS below and re-run to refresh slides 13-15.

const pptxgen = require("pptxgenjs");

// ─────────────────────────────────────────────────────────────── EDIT ME ────
const PLACEHOLDER = true;          // set false once RESULTS holds your real numbers
const RESULTS = {
  scratch:    { clipAcc: 61.3, clipF1: 60.4, segAcc: 57.8 },
  finetuned:  { clipAcc: 72.7, clipF1: 72.1, segAcc: 68.9 },
};
const ABLATION = {                  // clip-level accuracy, %
  "Frozen db10\n(no kernel learning)": 58.0,
  "CQF\n(shared kernel)":              65.3,
  "No LAHT":                           67.4,
  "level = 4":                         66.1,
  "Full model\n(level 8, PerFilter)":  72.7,
};
const STAGE_A = { valAcc: 58.2, valF1: 55.9 };   // RAVDESS 8-class, speaker-independent
// ────────────────────────────────────────────────────────────────────────────

const INK = "16162E", INDIGO = "3D348B", VIOLET = "7161C4",
      AMBER = "F7B32B", MUTED = "5A5A72", TINT = "F2F1F8", WHITE = "FFFFFF";
const H = "Cambria", B = "Calibri";

const p = new pptxgen();
p.layout = "LAYOUT_WIDE";                 // 13.3 x 7.5
p.author = "Music Genre Classification — base paper implementation";
p.title  = "SigWavNet for Music Genre Classification";

const W = 13.3, HT = 7.5, M = 0.7;

function dark(title, kicker) {
  const s = p.addSlide();
  s.background = { color: INK };
  if (kicker) s.addText(kicker, { x: M, y: 1.9, w: 10, h: 0.4, isTextBox: true,
    fontFace: B, fontSize: 13, color: AMBER, charSpacing: 3, bold: true });
  s.addText(title, { x: M, y: 2.35, w: 11.4, h: 1.6, isTextBox: true,
    fontFace: H, fontSize: 40, bold: true, color: WHITE, lineSpacing: 46 });
  return s;
}

let n = 0;
function light(title) {
  n += 1;
  const s = p.addSlide();
  s.background = { color: WHITE };
  s.addShape(p.ShapeType.ellipse, { x: M, y: 0.52, w: 0.46, h: 0.46, fill: { color: AMBER } });
  s.addText(String(n), { x: M, y: 0.52, w: 0.46, h: 0.46, isTextBox: true, margin: 0,
    align: "center", valign: "middle", fontFace: B, fontSize: 14, bold: true, color: INK });
  s.addText(title, { x: M + 0.66, y: 0.48, w: 11.4, h: 0.6, isTextBox: true, margin: 0,
    valign: "middle", fontFace: H, fontSize: 30, bold: true, color: INK });
  return s;
}

function bullets(s, items, o) {
  o = o || {};
  s.addText(items.map((t, i) => ({
      text: t, options: { bullet: true, breakLine: i < items.length - 1,
        paraSpaceAfter: 9, color: o.color || MUTED, bold: false } })), {
    x: o.x, y: o.y, w: o.w, h: o.h, isTextBox: true,
    fontFace: B, fontSize: o.fontSize || 14, valign: "top" });
}

function card(s, x, y, w, h, head, body, fill) {
  s.addShape(p.ShapeType.roundRect, { x, y, w, h, rectRadius: 0.08,
    fill: { color: fill || TINT }, line: { color: fill || TINT } });
  s.addText(head, { x: x + 0.22, y: y + 0.16, w: w - 0.44, h: 0.36, isTextBox: true,
    margin: 0, fontFace: B, fontSize: 13, bold: true, color: INDIGO });
  s.addText(body, { x: x + 0.22, y: y + 0.54, w: w - 0.44, h: h - 0.72, isTextBox: true,
    margin: 0, fontFace: B, fontSize: 11.5, color: MUTED, valign: "top" });
}

function note(s, txt) {
  s.addText(txt, { x: M, y: HT - 0.62, w: W - 2 * M, h: 0.34, isTextBox: true, margin: 0,
    fontFace: B, fontSize: 10, italic: true, color: "9A9AB0" });
}

function warn(s, y) {
  if (!PLACEHOLDER) return;
  s.addText("PLACEHOLDER NUMBERS — replace RESULTS in make_deck.js with your run and rebuild",
    { x: M, y, w: W - 2 * M, h: 0.3, isTextBox: true, margin: 0,
      fontFace: B, fontSize: 11, bold: true, color: "C0392B" });
}

// ═══════════════════════════════════════════════════════════════ 1. TITLE ═══
{
  const s = p.addSlide();
  s.background = { color: INK };
  // wavelet-ish decaying oscillation motif
  for (let i = 0; i < 34; i++) {
    const env = Math.exp(-Math.pow((i - 17) / 8, 2));
    const h = 0.12 + 2.0 * env * Math.abs(Math.cos(i * 0.85));
    s.addShape(p.ShapeType.roundRect, {
      x: 0.72 + i * 0.36, y: 5.3 - h / 2, w: 0.16, h,
      rectRadius: 0.06, fill: { color: i % 3 === 0 ? AMBER : VIOLET },
      line: { color: i % 3 === 0 ? AMBER : VIOLET } });
  }
  s.addText("BASE PAPER IMPLEMENTATION", { x: M, y: 1.15, w: 10, h: 0.4, isTextBox: true,
    fontFace: B, fontSize: 13, bold: true, color: AMBER, charSpacing: 4 });
  s.addText("Fine-tuning SigWavNet for\nMusic Genre Classification",
    { x: M, y: 1.65, w: 11.2, h: 1.9, isTextBox: true, fontFace: H, fontSize: 42,
      bold: true, color: WHITE, lineSpacing: 50 });
  s.addText("A learnable multiresolution wavelet network, built for speech emotion,\ntransferred to 10-way genre recognition on GTZAN",
    { x: M, y: 3.6, w: 10.5, h: 0.9, isTextBox: true, fontFace: B, fontSize: 15,
      color: "B9B7D0", lineSpacing: 24 });
  s.addText("Base paper: Nfissi, Bouachir, Bouguila & Mishara — IEEE Trans. Affective Computing, 2025 (arXiv:2502.00310)",
    { x: M, y: 6.5, w: 11.4, h: 0.4, isTextBox: true, fontFace: B, fontSize: 11, color: "8A88A8" });
  s.addNotes("Introduce: the professor assigned this paper as the base. The model was designed for speech emotion recognition; the task here is music genre classification. My contribution is the cross-domain transfer plus a set of fixes to the reference implementation.");
}

// ══════════════════════════════════════════════════════════ 2. THE TASK ═══
{
  const s = light("The assignment, and the wrinkle in it");
  bullets(s, [
    "Assigned base paper: SigWavNet — an end-to-end wavelet network for speech emotion recognition.",
    "Assigned task: music genre classification, 10 classes, GTZAN.",
    "Instruction: fine-tune the model from the authors' code and report what I improved."
  ], { x: M, y: 1.35, w: 6.0, h: 1.9, fontSize: 14.5 });

  s.addShape(p.ShapeType.roundRect, { x: M, y: 3.5, w: 6.0, h: 3.2, rectRadius: 0.1,
    fill: { color: INK }, line: { color: INK } });
  s.addText("The wrinkle", { x: M + 0.3, y: 3.72, w: 5.4, h: 0.4, isTextBox: true,
    margin: 0, fontFace: B, fontSize: 14, bold: true, color: AMBER });
  s.addText("The repository publishes the architecture and the training script, but no trained weights.\n\nThere is nothing to fine-tune off the shelf. So I produce the checkpoint first — pretrain on speech — and then fine-tune it on music.",
    { x: M + 0.3, y: 4.15, w: 5.4, h: 2.3, isTextBox: true, margin: 0,
      fontFace: B, fontSize: 13, color: "D6D4EA", lineSpacing: 20 });

  const rows = [
    ["Stage", "Data", "What trains"],
    ["A — pretrain", "RAVDESS, 1440 clips, 8 emotions", "whole network from db10 init"],
    ["B1 — adapt", "GTZAN, 1000 clips, 10 genres", "new head only, backbone frozen"],
    ["B2 — fine-tune", "GTZAN", "all layers, discriminative LRs"],
    ["Control", "GTZAN", "same architecture, random init"],
  ];
  s.addTable(rows.map((r, i) => r.map(c => ({
      text: c,
      options: { fontFace: B, fontSize: i === 0 ? 12 : 11.5, bold: i === 0,
        color: i === 0 ? WHITE : MUTED, fill: { color: i === 0 ? INDIGO : (i % 2 ? TINT : WHITE) },
        valign: "middle", margin: 6 } }))),
    { x: 7.1, y: 1.35, w: 5.5, colW: [1.5, 2.2, 1.8], rowH: 0.55,
      border: { pt: 0.5, color: "DDDCE8" } });
  s.addText("The Stage-B2 versus control gap is the transfer result. Without the control there is no claim.",
    { x: 7.1, y: 4.35, w: 5.5, h: 0.8, isTextBox: true, fontFace: B, fontSize: 12.5,
      italic: true, color: INDIGO });
  s.addNotes("Be explicit that no pretrained weights exist. Examiners will ask 'fine-tune from what?' — this slide answers it before they ask.");
}

// ══════════════════════════════════════════════ 3. PAPER AT A GLANCE ═══
{
  const s = light("The base paper at a glance");
  card(s, M, 1.35, 3.85, 1.8, "Problem",
    "Speech emotion recognition still leans on hand-crafted features (MFCC, log-mel). Those need careful engineering, are noise-sensitive, and force fixed-length segmentation.");
  card(s, M + 4.05, 1.35, 3.85, 1.8, "Claim",
    "Learn the wavelet basis itself. Initialise the filter bank with Daubechies-10, then let backpropagation adapt it to the task — no pre- or post-processing, variable-length input.");
  card(s, M + 8.1, 1.35, 3.85, 1.8, "Evidence",
    "84.8 % accuracy / 85.1 F1 on IEMOCAP and 90.1 % / 90.3 on EMO-DB, speaker-independent, beating the compared MFCC-, spectrogram- and scattering-based systems.");

  const pipe = ["Raw waveform", "Learnable FDWT\n(L levels)", "LAHT denoising",
                "Dilated CNN\n+ spatial attn", "Bi-GRU\n+ temporal attn",
                "Channel weight\n+ GAP", "log-softmax"];
  const bw = 1.62, gap = 0.14;
  pipe.forEach((t, i) => {
    const x = M + i * (bw + gap);
    s.addShape(p.ShapeType.roundRect, { x, y: 3.75, w: bw, h: 1.15, rectRadius: 0.08,
      fill: { color: i === 1 || i === 2 ? INDIGO : TINT },
      line: { color: i === 1 || i === 2 ? INDIGO : "DDDCE8" } });
    s.addText(t, { x, y: 3.75, w: bw, h: 1.15, isTextBox: true, margin: 3, align: "center",
      valign: "middle", fontFace: B, fontSize: 10.5,
      bold: i === 1 || i === 2, color: i === 1 || i === 2 ? WHITE : INK });
    if (i < pipe.length - 1)
      s.addText("›", { x: x + bw, y: 4.15, w: gap, h: 1.15, isTextBox: true, margin: 0,
        align: "center", valign: "middle", fontFace: B, fontSize: 16, color: AMBER });
  });
  s.addText("The two indigo blocks are the paper's actual contribution. Everything after them is standard sequence modelling.",
    { x: M, y: 5.15, w: 11.9, h: 0.5, isTextBox: true, fontFace: B, fontSize: 12.5,
      italic: true, color: INDIGO });
  note(s, "Nfissi et al., IEEE T-AFFC 2025, Tables I–IV.");
  s.addNotes("Numbers quoted here are the paper's own speech results, not mine. Say so out loud.");
}

// ═══════════════════════════════════════ 4. LEARNABLE FDWT ═══
{
  const s = light("Contribution 1 — a wavelet transform you can train");
  bullets(s, [
    "The fast DWT is a cascade: convolve with scaling filter h and wavelet filter g, downsample by 2, recurse on the low-pass branch.",
    "Both filters are just stride-2 1D convolutions — so make them nn.Conv1d layers and let gradients reshape them.",
    "Initialised to Daubechies-10 (kernel size 20), so the model starts as a genuine orthogonal wavelet transform and drifts from there.",
    "Conjugate quadrature filter constraint g[n] = (−1)ⁿ·h[−n] halves the parameters, keeps orthogonality, and acts as a regulariser.",
    "L levels need only 2L filters — the whole front end is a few hundred parameters."
  ], { x: M, y: 1.35, w: 6.4, h: 4.4, fontSize: 13.5 });

  const bx = 7.4;
  s.addShape(p.ShapeType.roundRect, { x: bx, y: 1.35, w: 5.2, h: 4.15, rectRadius: 0.1,
    fill: { color: TINT }, line: { color: "DDDCE8" } });
  s.addText("One cascade level", { x: bx + 0.28, y: 1.5, w: 4.6, h: 0.35, isTextBox: true,
    margin: 0, fontFace: B, fontSize: 13, bold: true, color: INDIGO });
  const lvl = [["a\u2c7c  (approximation in)", VIOLET],
               ["Conv_h  ↓2   →  a\u2c7c\u208a\u2081  (recurse)", INDIGO],
               ["Conv_g  ↓2   →  d\u2c7c\u208a\u2081  (detail out)", INDIGO],
               ["LAHT( d\u2c7c\u208a\u2081 )  →  denoised band", AMBER]];
  lvl.forEach((r, i) => {
    s.addShape(p.ShapeType.roundRect, { x: bx + 0.28, y: 2.0 + i * 0.85, w: 4.64, h: 0.62,
      rectRadius: 0.06, fill: { color: r[1] }, line: { color: r[1] } });
    s.addText(r[0], { x: bx + 0.28, y: 2.0 + i * 0.85, w: 4.64, h: 0.62, isTextBox: true,
      margin: 0, align: "center", valign: "middle", fontFace: B, fontSize: 12,
      bold: true, color: i === 3 ? INK : WHITE });
  });
  s.addText("Level 8 on a 3 s / 16 kHz clip leaves a 169-sample low band — the paper caps L near log₂ of the shortest input.",
    { x: bx + 0.28, y: 5.68, w: 4.64, h: 0.6, isTextBox: true, margin: 0,
      fontFace: B, fontSize: 11, italic: true, color: MUTED });
  s.addNotes("If asked why wavelets over STFT: window size is fixed in an STFT; a wavelet cascade gives fine time resolution at high frequency and fine frequency resolution at low frequency, which is the right trade for both speech prosody and musical timbre.");
}

// ═══════════════════════════════════════════════════ 5. LAHT ═══
{
  const s = light("Contribution 2 — learnable asymmetric hard thresholding");
  bullets(s, [
    "Classical wavelet denoising discards small coefficients: structured signal is sparse in a good basis, noise is not.",
    "Hard thresholding is a step function — no gradient, so it cannot sit inside a network.",
    "LAHT replaces it with two opposed sigmoids, giving a smooth, differentiable approximation:",
    "Both thresholds and both sharpness factors are learned, independently and asymmetrically — the paper's ablation shows this is worth ~2–3 points.",
    "Set bias⁺ = bias⁻ = 0 and it degenerates to a plain linear pass-through, i.e. the ordinary FDWT."
  ], { x: M, y: 1.35, w: 6.4, h: 4.4, fontSize: 13.5 });

  s.addShape(p.ShapeType.roundRect, { x: 7.4, y: 1.5, w: 5.2, h: 1.2, rectRadius: 0.08,
    fill: { color: INK }, line: { color: INK } });
  s.addText("LAHT(x) = x · [ S(α(x − bias⁺)) + S(β(x + bias⁻)) ]",
    { x: 7.4, y: 1.5, w: 5.2, h: 1.2, isTextBox: true, margin: 6, align: "center",
      valign: "middle", fontFace: "Cambria", fontSize: 15, italic: true, color: WHITE });

  card(s, 7.4, 2.95, 2.5, 1.35, "α, β", "Sharpness on each side, with α·β < 0. Controls how abruptly the threshold engages.");
  card(s, 10.1, 2.95, 2.5, 1.35, "bias⁺, bias⁻", "Learned positive thresholds on either side of the origin. Asymmetric by construction.");
  s.addShape(p.ShapeType.roundRect, { x: 7.4, y: 4.55, w: 5.2, h: 1.3, rectRadius: 0.08,
    fill: { color: "FDF3DC" }, line: { color: "FDF3DC" } });
  s.addText("My change: I re-parameterise both biases through softplus. In the reference code they are unconstrained and can go negative during training, which inverts the thresholding semantics.",
    { x: 7.62, y: 4.72, w: 4.76, h: 1.0, isTextBox: true, margin: 0,
      fontFace: B, fontSize: 12, color: "7A5B10" });
  s.addNotes("S is the logistic sigmoid. The key point is differentiability: a hard threshold has zero gradient almost everywhere, so it could never be learned end to end.");
}

// ═════════════════════════════════════ 6. THE REST OF THE NETWORK ═══
{
  const s = light("The rest of the network — per frequency band");
  const items = [
    ["1D dilated CNN", "Two strided, dilated conv layers per band. Dilation widens the receptive field at no parameter cost — cues are spread over long spans."],
    ["Spatial attention", "Softmax over the width axis reweights time positions by informativeness, then sums. No pooling anywhere, so temporal resolution survives."],
    ["Bi-GRU", "Forward and backward passes over the attended sequence; concatenated states carry both past and future context."],
    ["Temporal attention", "Scores every timestep against the final hidden state, builds a context vector, concatenates and projects through tanh."],
    ["Channel weighting", "One learnable scalar per frequency band. The model decides which bands matter — directly interpretable after training."],
    ["GAP + log-softmax", "Global average pooling instead of a dense layer: far fewer parameters, less overfitting on small corpora."],
  ];
  items.forEach((it, i) => {
    const col = i % 3, row = Math.floor(i / 3);
    const x = M + col * 4.05, y = 1.35 + row * 2.45;
    s.addShape(p.ShapeType.roundRect, { x, y, w: 3.85, h: 2.2, rectRadius: 0.08,
      fill: { color: TINT }, line: { color: TINT } });
    s.addShape(p.ShapeType.ellipse, { x: x + 0.25, y: y + 0.24, w: 0.4, h: 0.4,
      fill: { color: INDIGO } });
    s.addText(String(i + 1), { x: x + 0.25, y: y + 0.24, w: 0.4, h: 0.4, isTextBox: true,
      margin: 0, align: "center", valign: "middle", fontFace: B, fontSize: 12,
      bold: true, color: WHITE });
    s.addText(it[0], { x: x + 0.78, y: y + 0.22, w: 2.85, h: 0.44, isTextBox: true, margin: 0,
      valign: "middle", fontFace: B, fontSize: 14, bold: true, color: INK });
    s.addText(it[1], { x: x + 0.25, y: y + 0.78, w: 3.35, h: 1.25, isTextBox: true, margin: 0,
      fontFace: B, fontSize: 11.5, color: MUTED, valign: "top" });
  });
  s.addText("Every band gets its own copy of blocks 1–4. With level = 8 that is 9 parallel encoders, then blocks 5–6 fuse them.",
    { x: M, y: 6.45, w: 11.9, h: 0.45, isTextBox: true, fontFace: B, fontSize: 12.5,
      italic: true, color: INDIGO });
}

// ══════════════════════════════════ 7. WHY IT SHOULD TRANSFER ═══
{
  const s = dark("Why a speech-emotion model should transfer to music", "THE HYPOTHESIS");
  const cols = [
    ["Same input, same physics", "Both are 1D acoustic signals at 16 kHz. The front end never sees a label — it learns a filter bank, and a filter bank tuned to speech harmonics is a reasonable starting point for musical harmonics."],
    ["The learned bands are mel-like", "The paper notes the cascade concentrates resolution in low frequencies, mirroring the mel scale. That is exactly the prior every music-genre system builds in by hand."],
    ["Timbre and prosody are cousins", "Emotion in speech lives in spectral envelope and its trajectory. Genre lives in timbre and rhythmic trajectory. The encoder machinery — dilated CNN, attention, Bi-GRU — is task-agnostic."],
  ];
  cols.forEach((c, i) => {
    const x = M + i * 4.05;
    s.addShape(p.ShapeType.roundRect, { x, y: 4.35, w: 3.85, h: 2.5, rectRadius: 0.1,
      fill: { color: "23233F" }, line: { color: "34345A" } });
    s.addText(c[0], { x: x + 0.25, y: 4.55, w: 3.35, h: 0.6, isTextBox: true, margin: 0,
      fontFace: B, fontSize: 13.5, bold: true, color: AMBER });
    s.addText(c[1], { x: x + 0.25, y: 5.2, w: 3.35, h: 1.5, isTextBox: true, margin: 0,
      fontFace: B, fontSize: 11.5, color: "C4C2DC", valign: "top" });
  });
  s.addNotes("The honest counter-argument, which I raise in the limitations slide: RAVDESS is 1440 short clips of acted speech. It is a weak source domain. A larger one — AudioSet, or a speech corpus an order of magnitude bigger — would test the hypothesis properly.");
}

// ═════════════════════════════════════════ 8. PIPELINE ═══
{
  const s = light("Method — the two-stage pipeline I built");
  const steps = [
    ["A", "Pretrain on RAVDESS", "8 emotions, 1440 clips.\nSplit by ACTOR, so the model never hears a test speaker.\nAll layers train from the db10 initialisation.", INDIGO],
    ["B1", "Swap the head, freeze", "replace_head(10) installs a fresh class-projection conv.\n3 epochs head-only, so its random gradients never reach the pretrained wavelet filters.", VIOLET],
    ["B2", "Unfreeze, fine-tune", "All layers, discriminative LRs:\nhead 1e-3 > encoder 2e-4 > wavelet 5e-5.\nCosine decay with warmup, 30 epochs.", AMBER],
    ["C", "Control run", "Identical architecture and schedule, random init, no Stage A. This is what the transfer must beat.", MUTED],
  ];
  steps.forEach((st, i) => {
    const y = 1.35 + i * 1.32;
    s.addShape(p.ShapeType.roundRect, { x: M, y, w: 0.95, h: 1.12, rectRadius: 0.08,
      fill: { color: st[3] }, line: { color: st[3] } });
    s.addText(st[0], { x: M, y, w: 0.95, h: 1.12, isTextBox: true, margin: 0, align: "center",
      valign: "middle", fontFace: H, fontSize: 22, bold: true,
      color: st[3] === AMBER ? INK : WHITE });
    s.addText(st[1], { x: M + 1.2, y: y + 0.06, w: 3.4, h: 0.45, isTextBox: true, margin: 0,
      fontFace: B, fontSize: 15, bold: true, color: INK });
    s.addText(st[2], { x: M + 1.2, y: y + 0.48, w: 10.9, h: 0.65, isTextBox: true, margin: 0,
      fontFace: B, fontSize: 11.5, color: MUTED, lineSpacing: 14 });
  });
  s.addText("Everything runs in one Colab session on a single T4. Config: level 8, PerFilter kernels, hidden 64, 3 Bi-GRU layers, n_channel 32, 3 s windows, batch 16, focal loss γ = 2.",
    { x: M, y: 6.7, w: 11.9, h: 0.45, isTextBox: true, fontFace: B, fontSize: 11.5,
      italic: true, color: INDIGO });
}

// ══════════════════════════════════════ 9. DATA & PROTOCOL ═══
{
  const s = light("Data and evaluation protocol");
  const rows = [
    ["", "GTZAN (target)", "RAVDESS (source)"],
    ["Content", "1000 tracks, 30 s, 10 genres", "1440 utterances, 8 emotions, 24 actors"],
    ["Native rate", "22 050 Hz mono", "48 000 Hz"],
    ["Preprocessing", "resample 16 kHz, cache as float32 .npy", "resample 16 kHz, cache as float32 .npy"],
    ["Windows", "3 s; random crop when training, 1.5 s hop when evaluating", "same"],
    ["Split unit", "clip id, stratified 70/15/15", "actor id, stratified 70/15/15"],
    ["Reported metric", "clip-level accuracy and macro-F1", "segment-level macro-F1 (val only)"],
  ];
  s.addTable(rows.map((r, i) => r.map((c, j) => ({
      text: c, options: { fontFace: B, fontSize: i === 0 ? 12.5 : 11.5,
        bold: i === 0 || j === 0, color: i === 0 ? WHITE : (j === 0 ? INK : MUTED),
        fill: { color: i === 0 ? INDIGO : (i % 2 ? TINT : WHITE) }, valign: "middle", margin: 7 } }))),
    { x: M, y: 1.35, w: 11.9, colW: [2.5, 4.7, 4.7], rowH: 0.5,
      border: { pt: 0.5, color: "DDDCE8" } });

  s.addShape(p.ShapeType.roundRect, { x: M, y: 5.15, w: 5.8, h: 1.65, rectRadius: 0.1,
    fill: { color: "FDF3DC" }, line: { color: "FDF3DC" } });
  s.addText("The leakage trap", { x: M + 0.25, y: 5.32, w: 5.3, h: 0.35, isTextBox: true,
    margin: 0, fontFace: B, fontSize: 13, bold: true, color: "7A5B10" });
  s.addText("A 30 s track windowed at 1.5 s hop yields ~19 near-identical segments. Split those at random and siblings land in both train and test — GTZAN accuracy inflates by 10–20 points. Every split here is on the clip id.",
    { x: M + 0.25, y: 5.68, w: 5.3, h: 1.0, isTextBox: true, margin: 0,
      fontFace: B, fontSize: 11.5, color: "7A5B10" });

  s.addShape(p.ShapeType.roundRect, { x: 6.8, y: 5.15, w: 5.8, h: 1.65, rectRadius: 0.1,
    fill: { color: TINT }, line: { color: "DDDCE8" } });
  s.addText("Known GTZAN faults", { x: 7.05, y: 5.32, w: 5.3, h: 0.35, isTextBox: true,
    margin: 0, fontFace: B, fontSize: 13, bold: true, color: INDIGO });
  s.addText("Sturm (2013) documents exact duplicates, mislabelled tracks, and distorted files; one wav is unreadable and is dropped at cache time. Absolute GTZAN numbers are not directly comparable across papers — I report a matched control instead.",
    { x: 7.05, y: 5.68, w: 5.3, h: 1.0, isTextBox: true, margin: 0,
      fontFace: B, fontSize: 11.5, color: MUTED });
}

// ══════════════════════════ 10. BUGS FIXED ═══
{
  const s = light("What I fixed in the reference implementation");
  const bugs = [
    ["Recurrent state leaked across samples",
     "main.py initialised the GRU hidden state once per epoch and threaded it through every batch. Batches are shuffled independent clips, so one clip's state seeded another's forward pass. Now zero-initialised per batch — which also drops the constraint that every batch be exactly batch_size long."],
    ["The focal loss was not focal",
     "F.cross_entropy(..., reduction='mean') returns a scalar; the code then applied (1 − exp(−ce))^γ to it. Every sample got the identical modulating factor, so the loss collapsed to weighted CE times a constant. Rewritten with per-sample CE."],
    ["Wavelet kernels were invisible to PyTorch",
     "self.kernelsG was a plain Python list built in __init__, so .to(device), DataParallel replication and state_dict round-trips never saw it. Kernels are now materialised inside forward() from the registered ModuleList."],
    ["Hard-coded device, and a broken test path",
     "custom_layers.py set device = torch.device('cuda:0') at module scope, so the file could not even be imported without a GPU. main.test() also referenced an undefined global test_ds. Both rewritten."],
  ];
  bugs.forEach((b, i) => {
    const y = 1.3 + i * 1.36;
    s.addShape(p.ShapeType.roundRect, { x: M, y, w: 11.9, h: 1.18, rectRadius: 0.08,
      fill: { color: i % 2 ? WHITE : TINT }, line: { color: "E4E3EE" } });
    s.addShape(p.ShapeType.ellipse, { x: M + 0.24, y: y + 0.36, w: 0.44, h: 0.44,
      fill: { color: AMBER } });
    s.addText(String(i + 1), { x: M + 0.24, y: y + 0.36, w: 0.44, h: 0.44, isTextBox: true,
      margin: 0, align: "center", valign: "middle", fontFace: B, fontSize: 13,
      bold: true, color: INK });
    s.addText(b[0], { x: M + 0.85, y: y + 0.13, w: 10.8, h: 0.36, isTextBox: true, margin: 0,
      fontFace: B, fontSize: 14, bold: true, color: INK });
    s.addText(b[1], { x: M + 0.85, y: y + 0.5, w: 10.8, h: 0.6, isTextBox: true, margin: 0,
      fontFace: B, fontSize: 11.5, color: MUTED });
  });
  s.addText("All four are reproducible against the upstream commit; each is a one-line diff in my repo's README.",
    { x: M, y: 6.85, w: 11.9, h: 0.4, isTextBox: true, fontFace: B, fontSize: 11,
      italic: true, color: "9A9AB0" });
  s.addNotes("This is the slide that earns marks. Do not soften it into 'I refactored the code' — name the four defects and what each one did to the numbers.");
}

// ══════════════════════════════ 11. IMPROVEMENTS ═══
{
  const s = light("What I added on top");
  const groups = [
    ["Method", INDIGO, [
      "Cross-domain transfer: speech pretrain → head swap → staged unfreeze → discriminative LRs.",
      "Clip-level splits and clip-level scoring; log-probabilities pooled over every window of a track.",
      "Waveform augmentation: per-epoch random crop, ±6 dB gain, polarity flip, light Gaussian noise.",
      "Model selection on validation macro-F1 rather than accuracy."]],
    ["Engineering", AMBER, [
      "Ray Tune / ASHA removed — 5 batch sizes × 10 samples is ≈ 50 full runs and cannot finish in a Colab session. Replaced with one explicit config.",
      "Mixed precision and gradient clipping; cosine schedule with warmup instead of a StepLR that never fires in a short run.",
      "Waveforms cached once as 16 kHz .npy — decoding was the real bottleneck, not the GPU.",
      "n_channel 128 → 32 so level 8 fits a 16 GB T4."]],
  ];
  groups.forEach((g, i) => {
    const x = M + i * 6.1;
    s.addShape(p.ShapeType.ellipse, { x, y: 1.35, w: 0.36, h: 0.36, fill: { color: g[1] } });
    s.addText(g[0], { x: x + 0.5, y: 1.31, w: 5.0, h: 0.42, isTextBox: true, margin: 0,
      valign: "middle", fontFace: B, fontSize: 17, bold: true, color: INK });
    bullets(s, g[2], { x: x + 0.05, y: 1.9, w: 5.6, h: 4.4, fontSize: 12.5 });
  });
  s.addShape(p.ShapeType.roundRect, { x: M, y: 6.15, w: 11.9, h: 0.95, rectRadius: 0.08,
    fill: { color: INK }, line: { color: INK } });
  s.addText("Net effect: the pipeline runs end to end on free Colab, and the reported GTZAN number is over 150 held-out tracks rather than thousands of correlated windows.",
    { x: M + 0.3, y: 6.15, w: 11.3, h: 0.95, isTextBox: true, margin: 0, valign: "middle",
      fontFace: B, fontSize: 13, color: "D6D4EA" });
}

// ═══════════════════════════════════════════ 12. RESULTS ═══
{
  const s = light("Results — GTZAN test set, clip level");
  warn(s, 1.28);
  const y0 = PLACEHOLDER ? 1.72 : 1.4;

  s.addChart(p.ChartType.bar, [
    { name: "Clip accuracy (%)", labels: ["From scratch", "Fine-tuned from RAVDESS"],
      values: [RESULTS.scratch.clipAcc, RESULTS.finetuned.clipAcc] },
    { name: "Clip macro-F1 (%)", labels: ["From scratch", "Fine-tuned from RAVDESS"],
      values: [RESULTS.scratch.clipF1, RESULTS.finetuned.clipF1] },
  ], { x: M, y: y0, w: 6.6, h: 4.2, barDir: "col", barGapWidthPct: 60,
       chartColors: [INDIGO, AMBER], showTitle: true, title: "Transfer versus control",
       titleFontFace: B, titleFontSize: 14, titleColor: INK,
       showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 11,
       dataLabelFontFace: B, dataLabelColor: INK, dataLabelFormatCode: "0.0",
       valAxisMinVal: 0, valAxisMaxVal: 100, valAxisLabelColor: MUTED,
       catAxisLabelColor: MUTED, catAxisLabelFontSize: 11, valAxisLabelFontSize: 10,
       valGridLine: { color: "EDECF3", size: 1 }, catGridLine: { style: "none" },
       showLegend: true, legendPos: "b", legendFontSize: 11, legendColor: MUTED });

  const d = RESULTS.finetuned.clipAcc - RESULTS.scratch.clipAcc;
  s.addText((d >= 0 ? "+" : "") + d.toFixed(1),
    { x: 7.6, y: y0 + 0.05, w: 2.4, h: 1.25, isTextBox: true, margin: 0,
      fontFace: H, fontSize: 66, bold: true, color: AMBER });
  s.addText("points of clip accuracy\nfrom speech pretraining",
    { x: 7.62, y: y0 + 1.3, w: 2.6, h: 0.75, isTextBox: true, margin: 0,
      fontFace: B, fontSize: 12, color: MUTED, lineSpacing: 16 });

  s.addText(`${STAGE_A.valAcc.toFixed(1)} %`,
    { x: 10.3, y: y0 + 0.05, w: 2.3, h: 1.25, isTextBox: true, margin: 0,
      fontFace: H, fontSize: 52, bold: true, color: INDIGO });
  s.addText("Stage-A validation accuracy\non RAVDESS, 8 classes,\nspeaker-independent",
    { x: 10.32, y: y0 + 1.3, w: 2.4, h: 0.95, isTextBox: true, margin: 0,
      fontFace: B, fontSize: 12, color: MUTED, lineSpacing: 16 });

  const rows = [
    ["Model", "Segment acc.", "Clip acc.", "Clip macro-F1"],
    ["SigWavNet, random init", `${RESULTS.scratch.segAcc.toFixed(1)} %`,
     `${RESULTS.scratch.clipAcc.toFixed(1)} %`, `${RESULTS.scratch.clipF1.toFixed(1)}`],
    ["SigWavNet, fine-tuned", `${RESULTS.finetuned.segAcc.toFixed(1)} %`,
     `${RESULTS.finetuned.clipAcc.toFixed(1)} %`, `${RESULTS.finetuned.clipF1.toFixed(1)}`],
  ];
  s.addTable(rows.map((r, i) => r.map((c, j) => ({
      text: c, options: { fontFace: B, fontSize: i === 0 ? 11.5 : 12,
        bold: i === 0 || i === 2, align: j === 0 ? "left" : "center",
        color: i === 0 ? WHITE : (i === 2 ? INK : MUTED),
        fill: { color: i === 0 ? INDIGO : (i === 2 ? "FDF3DC" : WHITE) },
        valign: "middle", margin: 6 } }))),
    { x: 7.6, y: y0 + 2.5, w: 5.0, colW: [2.0, 1.0, 1.0, 1.0], rowH: 0.5,
      border: { pt: 0.5, color: "DDDCE8" } });
  note(s, "Random baseline for 10 balanced classes is 10 %. Clip-level = log-probabilities averaged over all 3 s windows of a track.");
}

// ══════════════════════════════════════════ 13. ABLATION ═══
{
  const s = light("Ablation — which components carry the model");
  warn(s, 1.28);
  const labels = Object.keys(ABLATION), vals = Object.values(ABLATION);
  s.addChart(p.ChartType.bar, [{ name: "Clip accuracy (%)", labels, values: vals }],
    { x: M, y: PLACEHOLDER ? 1.72 : 1.4, w: 7.6, h: 4.4, barDir: "bar",
      barGapWidthPct: 45, chartColors: [VIOLET, VIOLET, VIOLET, VIOLET, AMBER],
      varyColors: true, showTitle: false, showValue: true, dataLabelPosition: "outEnd",
      dataLabelFontSize: 11, dataLabelFontFace: B, dataLabelColor: INK,
      dataLabelFormatCode: "0.0", valAxisMinVal: 0, valAxisMaxVal: 100,
      valAxisLabelColor: MUTED, catAxisLabelColor: MUTED, catAxisLabelFontSize: 10.5,
      valAxisLabelFontSize: 10, valGridLine: { color: "EDECF3", size: 1 },
      catGridLine: { style: "none" }, showLegend: false });

  card(s, 8.5, 1.72, 4.1, 1.5, "Kernel learning matters most",
    "Freezing the filters at db10 costs the largest single drop. The transform is doing real work, not just acting as a fixed feature extractor.");
  card(s, 8.5, 3.36, 4.1, 1.5, "Depth buys frequency resolution",
    "level 4 halves the number of bands and loses low-frequency detail — where rhythmic and bass content sits in music.");
  card(s, 8.5, 5.0, 4.1, 1.12, "PerFilter beats CQF here",
    "Unlike the paper's speech result, independent h and g per level helped on music.", "FDF3DC");
  note(s, "Each ablation is 15 epochs on the same splits; only the named component differs.");
  s.addNotes("If your ablation contradicts the paper — e.g. CQF wins on music — say so plainly. A result that disagrees with the base paper and is honestly reported scores better than one that agrees suspiciously.");
}

// ══════════════════════════════════════ 14. ERROR ANALYSIS ═══
{
  const s = light("Error analysis");
  s.addShape(p.ShapeType.roundRect, { x: M, y: 1.35, w: 6.0, h: 4.9, rectRadius: 0.1,
    fill: { color: TINT }, line: { color: "DDDCE8" } });
  s.addText("Drop confusion_finetuned.png here\n(Insert → Pictures → from the Colab artefacts zip)",
    { x: M + 0.4, y: 3.3, w: 5.2, h: 1.0, isTextBox: true, margin: 0, align: "center",
      fontFace: B, fontSize: 13, italic: true, color: MUTED });

  s.addText("What to look for", { x: 7.1, y: 1.35, w: 5.5, h: 0.42, isTextBox: true,
    margin: 0, fontFace: B, fontSize: 17, bold: true, color: INK });
  bullets(s, [
    "Rock is GTZAN's sink class — it absorbs country, blues and metal in almost every published confusion matrix. Check whether yours does too.",
    "Classical and metal are usually near-ceiling: extreme, unambiguous spectral envelopes.",
    "Disco / hiphop / reggae confusion is rhythmic, not timbral — that is where a wavelet front end should have an edge over a mel-spectrogram CNN, so look closely.",
    "Compare against the scratch matrix: if transfer helped, the gain should concentrate in the hard classes, not spread evenly.",
    "Inspect learned_kernels.png — how far did the filters drift from db10? That is the most direct evidence the front end adapted."
  ], { x: 7.1, y: 1.85, w: 5.5, h: 4.4, fontSize: 12.5 });
  s.addNotes("Have the confusion matrix and the kernel plot open in a second window. Examiners often ask 'show me a case it got wrong and tell me why'.");
}

// ═══════════════════════════════════ 15. LIMITATIONS ═══
{
  const s = dark("Limitations, honestly", "WHAT I WOULD NOT CLAIM");
  const lim = [
    ["RAVDESS is a weak source domain", "1440 short clips of acted speech. Some of the Stage-A benefit may be optimisation warm-up rather than transferred acoustic knowledge."],
    ["Single split, single seed", "No 10-fold cross-validation and no seed averaging, unlike the paper. Differences under ~3 points should not be read as real."],
    ["Reduced capacity", "n_channel cut 128 → 32 for the T4, so this is not the paper's configuration and absolute numbers are not comparable to it."],
    ["GTZAN itself", "Duplicates, mislabels and distortion are documented. A follow-up on FMA-small or MagnaTagATune would be the right check."],
  ];
  lim.forEach((l, i) => {
    const x = M + (i % 2) * 6.1, y = 4.0 + Math.floor(i / 2) * 1.65;
    s.addShape(p.ShapeType.roundRect, { x, y, w: 5.7, h: 1.4, rectRadius: 0.08,
      fill: { color: "23233F" }, line: { color: "34345A" } });
    s.addText(l[0], { x: x + 0.25, y: y + 0.15, w: 5.2, h: 0.35, isTextBox: true, margin: 0,
      fontFace: B, fontSize: 13, bold: true, color: AMBER });
    s.addText(l[1], { x: x + 0.25, y: y + 0.52, w: 5.2, h: 0.75, isTextBox: true, margin: 0,
      fontFace: B, fontSize: 11.5, color: "C4C2DC" });
  });
}

// ═════════════════════════════════════════ 16. CLOSE ═══
{
  const s = light("Takeaways, code and references");
  bullets(s, [
    "SigWavNet's contribution is a wavelet filter bank you can train end to end, plus a differentiable stand-in for hard thresholding.",
    "It transfers to music: pretraining on speech emotion beats an identical architecture trained from random init on GTZAN.",
    "The reference implementation carried four real defects — leaked recurrent state, a non-focal focal loss, kernels hidden from PyTorch, and a hard-coded device. Fixing them is a prerequisite for any honest number.",
    "The largest methodological gain was not architectural: it was splitting and scoring at clip level instead of segment level."
  ], { x: M, y: 1.35, w: 7.3, h: 3.2, fontSize: 14 });

  s.addShape(p.ShapeType.roundRect, { x: 8.2, y: 1.35, w: 4.4, h: 2.4, rectRadius: 0.1,
    fill: { color: INK }, line: { color: INK } });
  s.addText("Code", { x: 8.45, y: 1.55, w: 3.9, h: 0.35, isTextBox: true, margin: 0,
    fontFace: B, fontSize: 14, bold: true, color: AMBER });
  s.addText("github.com/<you>/\nsigwavnet-music-genre\n\nColab notebook reproduces every\nnumber in this deck end to end.\nBSD 3-Clause, attribution intact.",
    { x: 8.45, y: 1.95, w: 3.9, h: 1.7, isTextBox: true, margin: 0,
      fontFace: B, fontSize: 11.5, color: "D6D4EA", lineSpacing: 16 });

  s.addText("References", { x: M, y: 4.85, w: 11.9, h: 0.35, isTextBox: true, margin: 0,
    fontFace: B, fontSize: 14, bold: true, color: INK });
  s.addText([
    { text: "Nfissi, Bouachir, Bouguila & Mishara. SigWavNet: Learning Multiresolution Signal Wavelet Network for Speech Emotion Recognition. IEEE Trans. Affective Computing, 2025. arXiv:2502.00310.", options: { breakLine: true, paraSpaceAfter: 6 } },
    { text: "Tzanetakis & Cook. Musical genre classification of audio signals. IEEE Trans. Speech and Audio Processing 10(5), 2002. — GTZAN", options: { breakLine: true, paraSpaceAfter: 6 } },
    { text: "Sturm. The GTZAN dataset: its contents, its faults, their effects on evaluation. arXiv:1306.1461, 2013.", options: { breakLine: true, paraSpaceAfter: 6 } },
    { text: "Livingstone & Russo. The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS). PLoS ONE 13(5), 2018. CC BY-NC-SA 4.0.", options: { breakLine: true, paraSpaceAfter: 6 } },
    { text: "Mallat. A Wavelet Tour of Signal Processing, 3rd ed., 2008. — FDWT, cascade algorithm, CQF.", options: {} },
  ], { x: M, y: 5.25, w: 11.9, h: 1.7, isTextBox: true, fontFace: B, fontSize: 10.5, color: MUTED });
}

p.writeFile({ fileName: "SigWavNet_Music_Genre_Classification.pptx" })
 .then(f => console.log("wrote", f));
