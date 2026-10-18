// Fig. 1 — plug-and-play compression framework for hybrid quantum RL policies (native, editable vector shapes)
const pptxgen = require("pptxgenjs");
const pres = new pptxgen();
pres.defineLayout({ name: "FIG", width: 7.2, height: 4.85 });
pres.layout = "FIG";
pres.title = "Fig. 1 LightQDRL framework";
const S = pres.shapes;
const F = "Arial";
const INK = "2B2B29", MUT = "6E6E68", LINE = "B9B9B3", BG = "F4F4F1";
const BLUE = "2A78D6", ORANGE = "EB6834", GREEN = "1BAF7A", PURPLE = "6250D6", AMBER = "C98A00";
const slide = pres.addSlide();
slide.background = { color: "FFFFFF" };

function txt(t, x, y, w, h, o = {}) {
  slide.addText(t, Object.assign({ x, y, w, h, fontFace: F, fontSize: 7, color: INK, margin: 0, valign: "middle",
    isTextBox: true }, o));
}
function box(x, y, w, h, fill, line, o = {}) {
  slide.addShape(o.round === false ? S.RECTANGLE : S.ROUNDED_RECTANGLE, Object.assign({ x, y, w, h,
    fill: { color: fill }, line: { color: line, width: o.lw || 0.75, dashType: o.dash || "solid" },
    rectRadius: o.r === undefined ? 0.06 : o.r }, o.extra || {}));
}
function arrow(x1, y1, x2, y2, color = MUT, w = 1) {
  slide.addShape(S.LINE, { x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1) || 0.001,
    h: Math.abs(y2 - y1) || 0.001, flipH: x2 < x1, flipV: y2 < y1,
    line: { color, width: w, endArrowType: "triangle" } });
}
function line(x1, y1, x2, y2, color = LINE, w = 0.75, dash = "solid") {
  slide.addShape(S.LINE, { x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1) || 0.001,
    h: Math.abs(y2 - y1) || 0.001, flipV: y2 < y1, line: { color, width: w, dashType: dash } });
}
function label(letter, x, y) { txt(letter, x, y, 0.2, 0.2, { fontSize: 11, bold: true, valign: "top" }); }
function badge(t, x, y, color) {
  box(x, y, 0.48, 0.17, color, color, { r: 0.08 });
  txt(t, x, y, 0.48, 0.17, { fontSize: 6, bold: true, color: "FFFFFF", align: "center" });
}

// ------------------------------------------------------------------ a: settings
label("a", 0.08, 0.06);
txt("Quantum-policy settings", 0.3, 0.08, 1.8, 0.18, { fontSize: 7.5, bold: true });
const cards = [["Quantum co-processor", "NISQ chip next to an edge server;\nshallow circuits, limited shots"],
               ["Quantum data", "states produced by quantum\nsensors or links feed the policy"],
               ["Future QPUs", "same policy class scales with\nhardware; cost still matters"]];
cards.forEach((c, i) => {
  const y = 0.36 + i * 0.52;
  box(0.12, y, 1.95, 0.44, BG, BG);
  slide.addShape(S.OVAL, { x: 0.19, y: y + 0.11, w: 0.22, h: 0.22, fill: { color: PURPLE }, line: { color: PURPLE } });
  txt(String(i + 1), 0.19, y + 0.11, 0.22, 0.22, { fontSize: 7, bold: true, color: "FFFFFF", align: "center" });
  txt(c[0], 0.48, y + 0.03, 1.55, 0.15, { fontSize: 7, bold: true });
  txt(c[1], 0.48, y + 0.18, 1.55, 0.24, { fontSize: 6, color: MUT, valign: "top" });
});
box(0.12, 1.94, 1.95, 0.3, "FFFFFF", PURPLE, { lw: 0.75 });
txt("Requirement: cheaper, deployable,\nnoise-robust quantum RL policies", 0.12, 1.94, 1.95, 0.3,
    { fontSize: 6.5, align: "center", color: PURPLE, bold: true });

// ------------------------------------------------------------------ b: hybrid policy + plug-in modules
label("b", 2.25, 0.06);
txt("Hybrid policy backbone shared by all agents, with plug-in compression", 2.47, 0.08, 4.6, 0.18,
    { fontSize: 7.5, bold: true });
// observation
box(2.3, 0.75, 0.55, 0.9, BG, LINE);
txt([{ text: "State s", options: {} }, { text: "t", options: { subscript: true } }], 2.3, 0.8, 0.55, 0.15, { fontSize: 6.5, bold: true, align: "center" });
txt("channels\nqueues\narrivals\n(30-dim)", 2.3, 0.97, 0.55, 0.62, { fontSize: 5.8, color: MUT, align: "center", valign: "top" });
arrow(2.86, 1.2, 2.98, 1.2);
// encoder
box(3.0, 0.62, 0.78, 1.16, "EAF1FB", BLUE);
txt("Classical encoder", 3.0, 0.66, 0.78, 0.15, { fontSize: 6, bold: true, align: "center", color: BLUE });
["FC 64", "LayerNorm·ReLU", "FC 6", "π·tanh → angles"].forEach((t, i) => {
  box(3.07, 0.86 + i * 0.21, 0.64, 0.17, "FFFFFF", "C9DBF3", { r: 0.03 });
  txt(t, 3.07, 0.86 + i * 0.21, 0.64, 0.17, { fontSize: 5.6, align: "center" });
});
badge("JCQQ", 3.15, 0.42, ORANGE);
arrow(3.79, 1.2, 3.9, 1.2);
// circuit
const cx = 3.93, cy = 0.66, cw = 1.72, ch = 1.1;
box(cx - 0.03, cy - 0.06, cw + 0.06, ch + 0.12, "F1EFFB", PURPLE);
const nq = 6, dy = ch / (nq - 1) * 0.92;
const ys = [...Array(nq).keys()].map(q => cy + 0.06 + q * dy * 0.98);
ys.forEach(y => line(cx + 0.02, y, cx + cw - 0.02, y, "8E86C9", 0.6));
const gw = 0.17, gh = 0.13;
function gate(x, y, t, on = true) {
  box(x, y - gh / 2, gw, gh, on ? "FFFFFF" : "F1EFFB", on ? PURPLE : "B8B3DC", { r: 0.02, lw: 0.6, dash: on ? "solid" : "dash" });
  txt(t, x, y - gh / 2, gw, gh, { fontSize: 5, align: "center", color: on ? PURPLE : "B8B3DC" });
}
// pruned pattern (illustrative): which rotation blocks / CNOTs are removed by DQCP
const prunedR = [[1, 0, 0, 1, 0, 1], [0, 1, 1, 0, 1, 0]];
const prunedC = [[0, 1, 0, 1, 1, 0], [1, 0, 1, 0, 0, 1]];
ys.forEach((y, q) => gate(cx + 0.05, y, "Ry", true));
for (let l = 0; l < 2; l++) {
  const xr = cx + 0.3 + l * 0.72;
  ys.forEach((y, q) => gate(xr, y, "R", !prunedR[l][q]));
  for (let q = 0; q < nq; q++) {
    const xc = xr + 0.25 + q * 0.06;
    const t = (q + 1) % nq;
    const on = !prunedC[l][q];
    const col = on ? PURPLE : "C9C5E6";
    const y1 = ys[q], y2 = ys[t];
    line(xc, Math.min(y1, y2), xc, Math.max(y1, y2), col, 0.6, on ? "solid" : "dash");
    slide.addShape(S.OVAL, { x: xc - 0.02, y: y1 - 0.02, w: 0.04, h: 0.04, fill: { color: col }, line: { color: col } });
    slide.addShape(S.OVAL, { x: xc - 0.035, y: y2 - 0.035, w: 0.07, h: 0.07, fill: { color: "FFFFFF" }, line: { color: col, width: 0.6 } });
  }
}
txt("6-qubit VQC, 2 layers", cx, cy + ch + 0.08, cw, 0.14, { fontSize: 6, color: PURPLE, align: "center", bold: true });
txt("dashed = removed by DQCP", cx, cy + ch + 0.2, cw, 0.12, { fontSize: 5.6, color: MUT, align: "center" });
badge("DQCP", cx + 0.35, 0.42, ORANGE);
badge("JCQQ", cx + 0.9, 0.42, ORANGE);
arrow(cx + cw + 0.04, 1.2, cx + cw + 0.15, 1.2);
// readout + heads
const hx = 5.83;
box(hx, 0.62, 0.62, 1.16, "EAF1FB", BLUE);
txt("⟨Z⟩ readout", hx, 0.66, 0.62, 0.15, { fontSize: 6.3, bold: true, align: "center", color: BLUE });
["Q head", "A + V heads", "π head"].forEach((t, i) => {
  box(hx + 0.06, 0.9 + i * 0.27, 0.5, 0.21, "FFFFFF", "C9DBF3", { r: 0.03 });
  txt(t, hx + 0.06, 0.9 + i * 0.27, 0.5, 0.21, { fontSize: 5.6, align: "center" });
});
badge("PEQA", hx + 0.07, 0.42, GREEN);
arrow(hx + 0.63, 1.2, hx + 0.74, 1.2);
box(6.6, 0.75, 0.52, 0.9, BG, LINE);
txt([{ text: "Action a", options: {} }, { text: "t", options: { subscript: true } }], 6.6, 0.8, 0.52, 0.15, { fontSize: 6.5, bold: true, align: "center" });
txt("10 binary\nbranches\n(offload /\nlocal)", 6.6, 0.97, 0.52, 0.62, { fontSize: 5.8, color: MUT, align: "center", valign: "top" });
// module legend row
const mods = [["DQCP", ORANGE, "Dynamic quantum-circuit pruning. EMA saliency of rotations, loss sensitivity of CNOTs, exponential schedule; 50% of both removed during training"],
              ["JCQQ", ORANGE, "Joint classical–quantum quantization. INT8 weights and activations (straight-through), 8-bit phase grid for all angles, 8-bit ⟨Z⟩"],
              ["PEQA", GREEN, "Parameter-efficient adaptation after deployment. Rank-2 adapters, offsets of retained angles, two scales (≈8% of parameters)"]];
mods.forEach((m, i) => {
  const y = 2.0 + i * 0.0;
  const x = 2.3 + i * 1.62;
  box(x, 2.12, 1.55, 0.62, "FFFFFF", m[1], { lw: 0.75 });
  txt(m[0], x + 0.06, 2.15, 0.5, 0.14, { fontSize: 6.8, bold: true, color: m[1] });
  txt(m[2], x + 0.06, 2.3, 1.44, 0.42, { fontSize: 5.4, color: INK, valign: "top" });
});

// ------------------------------------------------------------------ c: algorithm-agnostic benchmark protocol
label("c", 0.08, 2.84);
txt("Algorithm-agnostic evaluation protocol", 0.3, 2.86, 4.0, 0.18, { fontSize: 7.5, bold: true });
const algos = ["Q-DQN", "Q-DDQN", "Q-Dueling", "Q-A2C", "Q-A3C", "Q-PPO", "Q-TRPO"];
algos.forEach((a, i) => {
  const x = 0.14 + (i % 4) * 0.62, y = 3.16 + Math.floor(i / 4) * 0.27;
  box(x, y, 0.57, 0.21, i < 3 ? "EAF1FB" : "F1EFFB", i < 3 ? BLUE : PURPLE, { r: 0.1, lw: 0.6 });
  txt(a, x, y, 0.57, 0.21, { fontSize: 5.6, align: "center" });
});
txt("value-based", 0.14, 3.73, 1.2, 0.13, { fontSize: 5.5, color: BLUE });
txt("policy-gradient", 1.38, 3.73, 1.2, 0.13, { fontSize: 5.5, color: PURPLE });
arrow(2.62, 3.44, 2.8, 3.44);
const steps = [["Train base & light", "same seed and update rule;\n3 tasks × 10 seeds"],
               ["Selection", "pre-registered, validation\nseeds: light ≥ base − δ"],
               ["LightQDRL-X", "recommended configuration\n+ 2³ factorial ablation"],
               ["Deployment tests", "noise · shots · topology ·\nshift adaptation · QPU"]];
steps.forEach((s, i) => {
  const x = 2.82 + i * 1.08;
  const col = i === 2 ? ORANGE : (i === 3 ? PURPLE : INK);
  box(x, 3.14, 0.98, 0.62, i === 2 ? "FDEFE8" : BG, i === 2 ? ORANGE : LINE, { lw: 0.75 });
  txt(s[0], x + 0.04, 3.17, 0.92, 0.16, { fontSize: 6.3, bold: true, color: col, align: "center" });
  txt(s[1], x + 0.04, 3.35, 0.92, 0.38, { fontSize: 5.4, color: MUT, align: "center", valign: "top" });
  if (i < 3) arrow(x + 0.98, 3.44, x + 1.08, 3.44);
});
// environments strip
const envs = [["WP-MEC (main)", "queue-aware offloading driven by\nindustrial video tasks, 10 devices", BLUE],
              ["Multichannel access", "8 Gilbert–Elliott channels,\npartially observed", GREEN],
              ["CartPole-v1", "standard control\nbenchmark", AMBER]];
envs.forEach((e, i) => {
  const x = 0.14 + i * 1.58;
  box(x, 4.02, 1.5, 0.48, "FFFFFF", e[2], { lw: 0.75 });
  txt(e[0], x + 0.06, 4.05, 1.4, 0.15, { fontSize: 6.3, bold: true, color: e[2] });
  txt(e[1], x + 0.06, 4.2, 1.4, 0.28, { fontSize: 5.4, color: MUT, valign: "top" });
});
box(4.9, 4.02, 2.2, 0.48, "F1EFFB", PURPLE, { lw: 0.75 });
txt("Hardware check", 4.96, 4.05, 2.1, 0.15, { fontSize: 6.3, bold: true, color: PURPLE });
txt("Origin Quantum WK_C180 superconducting\nprocessor: base vs light circuits on identical states", 4.96, 4.2, 2.1, 0.28,
    { fontSize: 5.4, color: MUT, valign: "top" });
txt("Environments", 0.14, 4.54, 2.0, 0.14, { fontSize: 5.5, color: MUT });

pres.writeFile({ fileName: process.argv[2] || "fig1.pptx" }).then(f => console.log("wrote", f));
