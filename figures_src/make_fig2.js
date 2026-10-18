// Fig. 2 — evaluation environments: queue-aware WP-MEC driven by industrial video tasks, and multichannel access
const pptxgen = require("pptxgenjs");
const D = require("./fig2_data.json");
const pres = new pptxgen();
pres.defineLayout({ name: "FIG", width: 7.2, height: 5.0 });
pres.layout = "FIG";
pres.title = "Fig. 2 Environments";
const S = pres.shapes;
const F = "Arial";
const INK = "2B2B29", MUT = "6E6E68", LINE = "B9B9B3", BG = "F4F4F1", GRID = "E6E6E2";
const BLUE = "2A78D6", ORANGE = "EB6834", GREEN = "1BAF7A", PURPLE = "6250D6", AMBER = "EDA100", RED = "E34948";
const slide = pres.addSlide();
slide.background = { color: "FFFFFF" };

function txt(t, x, y, w, h, o = {}) {
  slide.addText(t, Object.assign({ x, y, w, h, fontFace: F, fontSize: 7, color: INK, margin: 0, valign: "middle",
    isTextBox: true }, o));
}
function box(x, y, w, h, fill, line, o = {}) {
  slide.addShape(o.rect ? S.RECTANGLE : S.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: fill },
    line: { color: line, width: o.lw || 0.75, dashType: o.dash || "solid" }, rectRadius: o.r === undefined ? 0.05 : o.r });
}
function seg(x1, y1, x2, y2, color, w = 0.75, o = {}) {
  slide.addShape(S.LINE, { x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1) || 0.001,
    h: Math.abs(y2 - y1) || 0.001, flipH: x2 < x1, flipV: y2 < y1,
    line: Object.assign({ color, width: w, dashType: o.dash || "solid" }, o.arrow ? { endArrowType: "triangle" } : {}) });
}
function label(letter, x, y) { txt(letter, x, y, 0.2, 0.2, { fontSize: 11, bold: true, valign: "top" }); }
function title(t, x, y, w) { txt(t, x, y, w, 0.18, { fontSize: 7.5, bold: true }); }
const axis = { catAxisLabelColor: MUT, valAxisLabelColor: MUT, catAxisLabelFontSize: 6, valAxisLabelFontSize: 6,
  catAxisLabelFontFace: F, valAxisLabelFontFace: F, valGridLine: { color: GRID, size: 0.5 }, catGridLine: { style: "none" },
  catAxisLineColor: LINE, valAxisLineShow: false, showValAxisTitle: true, valAxisTitleFontSize: 6, valAxisTitleColor: MUT,
  valAxisTitleFontFace: F, showCatAxisTitle: true, catAxisTitleFontSize: 6, catAxisTitleColor: MUT, catAxisTitleFontFace: F,
  legendFontSize: 6, legendFontFace: F, legendColor: INK, catAxisMajorTickMark: "none", valAxisMajorTickMark: "none" };

// ------------------------------------------------------------------ a: WP-MEC system
label("a", 0.08, 0.06);
title("Queue-aware wireless-powered MEC (main task)", 0.3, 0.08, 3.3);
// access point + edge server
const apx = 0.35, apy = 0.95;
box(apx, apy, 0.62, 0.62, "EAF1FB", BLUE);
txt("AP +\nedge\nserver", apx, apy, 0.62, 0.62, { fontSize: 6.3, bold: true, align: "center", color: BLUE });
seg(apx + 0.31, apy, apx + 0.31, apy - 0.22, BLUE, 1);
slide.addShape(S.ISOSCELES_TRIANGLE, { x: apx + 0.23, y: apy - 0.36, w: 0.16, h: 0.14, fill: { color: BLUE }, line: { color: BLUE } });
// devices
const dev = [];
for (let i = 0; i < 10; i++) {
  const x = 1.55 + (i % 5) * 0.4, y = 0.48 + Math.floor(i / 5) * 0.86;
  dev.push([x, y]);
  const off = [0, 1, 4, 5].includes(i);
  box(x, y, 0.3, 0.22, "FFFFFF", off ? ORANGE : MUT, { lw: 0.75, r: 0.04 });
  slide.addShape(S.OVAL, { x: x + 0.1, y: y + 0.06, w: 0.1, h: 0.1, fill: { color: off ? ORANGE : MUT }, line: { color: "FFFFFF", width: 0.5 } });
  // queue bar
  const q = [0.7, 0.35, 0.2, 0.55, 0.9, 0.3, 0.65, 0.15, 0.45, 0.25][i];
  box(x + 0.32, y, 0.05, 0.22, "FFFFFF", LINE, { rect: true, lw: 0.5 });
  box(x + 0.32, y + 0.22 * (1 - q), 0.05, 0.22 * q, BLUE, BLUE, { rect: true, lw: 0.25 });
  txt(String(i + 1), x, y + 0.23, 0.3, 0.11, { fontSize: 5, color: MUT, align: "center" });
}
// links to device 1 (offload) and device 3 (local): energy (dashed amber) + data (orange)
seg(apx + 0.63, apy + 0.2, dev[0][0], dev[0][1] + 0.11, AMBER, 0.75, { dash: "dash", arrow: true });
seg(dev[5][0], dev[5][1] + 0.11, apx + 0.63, apy + 0.45, ORANGE, 0.9, { arrow: true });
txt("energy (WPT)", 0.72, 0.5, 0.7, 0.12, { fontSize: 5.5, color: AMBER });
txt("task bits (TDMA)", 0.98, 1.6, 0.8, 0.12, { fontSize: 5.5, color: ORANGE });
// legend
box(1.55, 2.0, 0.14, 0.1, "FFFFFF", ORANGE, { r: 0.02 }); txt("offloads", 1.72, 1.99, 0.5, 0.12, { fontSize: 5.5 });
box(2.22, 2.0, 0.14, 0.1, "FFFFFF", MUT, { r: 0.02 }); txt("computes locally", 2.39, 1.99, 0.8, 0.12, { fontSize: 5.5 });
box(3.12, 2.0, 0.05, 0.1, BLUE, BLUE, { rect: true }); txt("queue", 3.2, 1.99, 0.4, 0.12, { fontSize: 5.5 });
// frame structure
txt("Frame (1 s)", 0.3, 2.22, 0.6, 0.12, { fontSize: 6, bold: true });
const fx = 0.3, fy = 2.36, fw = 3.3, fh = 0.2;
box(fx, fy, fw * 0.3, fh, "FFF4D6", AMBER, { rect: true, lw: 0.6 });
txt("WPT  a = 0.3", fx, fy, fw * 0.3, fh, { fontSize: 5.8, align: "center" });
["τ₁", "τ₂", "τ₅", "τ₆"].forEach((t, k) => {
  const w = fw * 0.7 / 4;
  box(fx + fw * 0.3 + k * w, fy, w, fh, "FDEFE8", ORANGE, { rect: true, lw: 0.6 });
  txt(t + " = (1−a)/n_off", fx + fw * 0.3 + k * w, fy, w, fh, { fontSize: 5.2, align: "center" });
});
txt("Decision m ∈ {0,1}¹⁰ per frame; reward = Σᵢ wᵢ·servedᵢ − κ·dropped (Lyapunov drift-plus-penalty), wᵢ ∝ Qᵢ",
    0.3, 2.6, 3.4, 0.24, { fontSize: 5.6, color: MUT, valign: "top" });

// ------------------------------------------------------------------ b: video task sizes
label("b", 3.72, 0.06);
title("Industrial video tasks", 3.94, 0.08, 1.6);
slide.addChart(pres.charts.BAR, [{ name: "clips", labels: D.hist.labels, values: D.hist.counts }], Object.assign({}, axis, {
  x: 3.72, y: 0.26, w: 1.72, h: 1.55, barDir: "col", chartColors: [BLUE], barGapWidthPct: 25, showLegend: false,
  valAxisTitle: "Clips", catAxisTitle: "Clip size (MB)", catAxisLabelRotate: 315,
  showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 5.5, dataLabelColor: INK, dataLabelFontFace: F }));
txt(`${D.hist.n} clips from a construction-monitoring network; median ${D.hist.median.toFixed(1)} MB (256 MB = segment cap)`,
    3.74, 1.8, 1.72, 0.26, { fontSize: 5.3, color: MUT, valign: "top" });

// ------------------------------------------------------------------ c: arrivals of three devices
label("c", 5.5, 0.06);
title("Bursty arrivals", 5.72, 0.08, 1.4);
const T = [...Array(60).keys()].map(t => String(t));
const arr = [0, 1, 2].map(i => ({ name: `device ${i + 1}`, labels: T, values: D.arrivals_Mbit.slice(0, 60).map(r => r[i]) }));
slide.addChart(pres.charts.BAR, arr, Object.assign({}, axis, {
  x: 5.48, y: 0.26, w: 1.68, h: 1.55, barDir: "col", barGrouping: "stacked", chartColors: [BLUE, ORANGE, GREEN],
  barGapWidthPct: 40, showLegend: true, legendPos: "t", valAxisTitle: "Arrival (Mbit)", catAxisTitle: "Frame",
  catAxisLabelFrequency: 20, catAxisLabelRotate: 0 }));
txt("p = 0.3 per frame; device rate λᵢ = 0.5 × its service share", 5.5, 1.8, 1.66, 0.26, { fontSize: 5.3, color: MUT, valign: "top" });

// ------------------------------------------------------------------ d: channel fading
label("d", 3.72, 2.12);
title("Correlated fading", 3.94, 2.14, 1.6);
const T2 = [...Array(100).keys()].map(t => String(t));
const ch = [0, 1, 2].map(i => ({ name: `device ${i + 1}`, labels: T2, values: D.h_norm.map(r => r[i]) }));
slide.addChart(pres.charts.LINE, ch, Object.assign({}, axis, {
  x: 3.72, y: 2.3, w: 1.72, h: 1.32, chartColors: [BLUE, ORANGE, GREEN], lineSize: 1, lineDataSymbol: "none",
  showLegend: false, valAxisTitle: "hᵢ / gᵢ", catAxisTitle: "Frame", catAxisLabelFrequency: 25, catAxisLabelRotate: 0 }));
txt("Gauss–Markov, ρ = 0.95; gᵢ from the DROO dataset", 3.74, 3.6, 1.72, 0.2, { fontSize: 5.3, color: MUT, valign: "top" });

// ------------------------------------------------------------------ e: reference policies, nominal and shifted
label("e", 5.5, 2.12);
title("Reference policies", 5.72, 2.14, 1.4);
const kinds = ["random", "local", "offload", "maxrate", "myopic"];
const kl = ["Random", "All local", "All offload", "Max-rate", "Myopic MW"];
const ser = [["nominal", "Nominal"], ["traffic", "Traffic ×1.5"], ["channel", "Channel −3 dB"]].map(([k, n]) => (
  { name: n, labels: kl, values: kinds.map(q => Math.round(D.refs[k][q])) }));
slide.addChart(pres.charts.BAR, ser, Object.assign({}, axis, {
  x: 5.48, y: 2.3, w: 1.68, h: 1.32, barDir: "col", chartColors: [BLUE, ORANGE, PURPLE], barGapWidthPct: 30,
  showLegend: true, legendPos: "t", valAxisTitle: "Episode return", showCatAxisTitle: false, catAxisLabelRotate: 315 }));
txt("Scores are normalized: 0 = random, 1 = myopic max-weight (exhaustive 2¹⁰ search)", 5.5, 3.6, 1.66, 0.2,
    { fontSize: 5.3, color: MUT, valign: "top" });

// ------------------------------------------------------------------ f: multichannel access
label("f", 0.08, 2.92);
title("Dynamic multichannel access (second task)", 0.3, 2.94, 3.3);
for (let c = 0; c < 8; c++) {
  const y = 3.18 + c * 0.2;
  txt(`ch ${c + 1}`, 0.3, y, 0.3, 0.16, { fontSize: 5.5, color: MUT });
  const pat = [1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1];
  for (let t = 0; t < 16; t++) {
    const good = pat[(t + 3 * c) % 16] ^ (c % 3 === 0 ? 0 : (t > 10 ? 1 : 0));
    const chosen = (c === 4 && t < 6) || (c === 1 && t >= 6 && t < 11) || (c === 6 && t >= 11);
    box(0.62 + t * 0.105, y + 0.02, 0.095, 0.13, good ? "D7F1E6" : "F6D9D6", chosen ? INK : "FFFFFF", { rect: true, lw: chosen ? 0.9 : 0.4 });
  }
}
txt("time slot →", 0.62, 4.79, 1.7, 0.12, { fontSize: 5.5, color: MUT });
// two-state chain
const sx = 2.5, sy = 3.35;
slide.addShape(S.OVAL, { x: sx, y: sy, w: 0.42, h: 0.42, fill: { color: "D7F1E6" }, line: { color: GREEN, width: 0.75 } });
txt("good", sx, sy, 0.42, 0.42, { fontSize: 6, align: "center", color: GREEN, bold: true });
slide.addShape(S.OVAL, { x: sx + 0.75, y: sy, w: 0.42, h: 0.42, fill: { color: "F6D9D6" }, line: { color: RED, width: 0.75 } });
txt("bad", sx + 0.75, sy, 0.42, 0.42, { fontSize: 6, align: "center", color: RED, bold: true });
seg(sx + 0.43, sy + 0.12, sx + 0.74, sy + 0.12, MUT, 0.75, { arrow: true });
seg(sx + 0.74, sy + 0.3, sx + 0.43, sy + 0.3, MUT, 0.75, { arrow: true });
txt("p_gb", sx + 0.45, sy - 0.04, 0.28, 0.12, { fontSize: 5.3, color: MUT, align: "center" });
txt("p_bg", sx + 0.45, sy + 0.32, 0.28, 0.12, { fontSize: 5.3, color: MUT, align: "center" });
txt("8 Gilbert–Elliott channels with heterogeneous quality and memory; only the chosen channel is observed (outlined). Reward +1 / −1.",
    2.38, 3.86, 1.3, 0.5, { fontSize: 5.4, color: MUT, valign: "top" });
box(2.38, 4.42, 1.3, 0.38, BG, BG);
txt([{ text: "Random  10.7", options: { breakLine: true } }, { text: "Belief-myopic  89.2", options: { bold: true } }],
    2.44, 4.44, 1.2, 0.34, { fontSize: 6, valign: "middle" });

// ------------------------------------------------------------------ g: shifts used for adaptation
label("g", 3.72, 3.88);
title("Distribution shifts for adaptation", 3.94, 3.9, 3.0);
box(3.76, 4.14, 1.66, 0.46, "FDEFE8", ORANGE);
txt([{ text: "Video-traffic surge", options: { bold: true, color: ORANGE, breakLine: true } },
     { text: "every device's arrival rate ×1.5 (load 0.5 → 0.75)", options: {} }], 3.82, 4.17, 1.56, 0.4,
    { fontSize: 5.8, valign: "top" });
box(5.5, 4.14, 1.66, 0.46, "F1EFFB", PURPLE);
txt([{ text: "Channel deterioration", options: { bold: true, color: PURPLE, breakLine: true } },
     { text: "mean gain of every device ×0.5 (−3 dB); arrivals unchanged, effective load rises", options: {} }],
    5.56, 4.17, 1.56, 0.4, { fontSize: 5.8, valign: "top" });

pres.writeFile({ fileName: process.argv[2] || "fig2.pptx" }).then(f => console.log("wrote", f));
