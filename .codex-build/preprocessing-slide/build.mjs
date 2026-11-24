import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const workspaceDir = "/Users/leonakryeziu/ZHAW_BSc_Modul/SEM6/BA/spectroscopy-qml-xkye-verdiant";
const buildDir = path.join(workspaceDir, ".codex-build/preprocessing-slide");
const outputDir = path.join(workspaceDir, "outputs/preprocessing-slide-20261029");
const SKILL_DIR = "/Users/leonakryeziu/.codex/plugins/cache/openai-primary-runtime/presentations/26.1025.11240/skills/presentations";
const RUNTIME_PYTHON = "/Users/leonakryeziu/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3";
const FINAL_PPTX = path.join(outputDir, "preprocessing_selection_slide_final.pptx");
const fontFamily = "Aptos";

await fs.mkdir(buildDir, { recursive: true });
await fs.mkdir(outputDir, { recursive: true });

const deck = Presentation.create({ slideSize: { width: 1280, height: 720 } });
const slide = deck.slides.add();
slide.background.fill = "#F7F6F2";

function addText(name, text, position, style = {}, fill = "none", line = { style: "solid", fill: "none", width: 0 }) {
  const shape = slide.shapes.add({ geometry: "textbox", name, position, fill, line });
  shape.text = text;
  shape.text.style = { fontFamily, color: "#152633", fontSize: 18, ...style };
  return shape;
}

function addRect(name, position, fill, radius = false, lineFill = "none") {
  return slide.shapes.add({
    geometry: radius ? "roundRect" : "rect",
    name,
    position,
    fill,
    line: { style: "solid", fill: lineFill, width: lineFill === "none" ? 0 : 1 },
  });
}

addText(
  "slide-title",
  "Auswahl des Preprocessings",
  { left: 64, top: 42, width: 760, height: 48 },
  { fontSize: 34, bold: true, color: "#102B3F" },
);
addText(
  "slide-subtitle",
  "Eine gemeinsame Pipeline reduziert technische Unterschiede zwischen den Spektren",
  { left: 64, top: 94, width: 830, height: 32 },
  { fontSize: 18, color: "#5C6972" },
);
addText(
  "screening-label",
  "SCREENING: 15 KONFIGURATIONEN",
  { left: 930, top: 52, width: 286, height: 28 },
  { fontSize: 13, bold: true, color: "#2E6F7E", alignment: "right" },
);

addRect("title-rule", { left: 64, top: 140, width: 1152, height: 2 }, "#1C5264");

const steps = [
  {
    number: "01",
    title: "FIR Resampling",
    target: "Raster",
    body: "Vereinheitlicht das Wellenzahlraster und reduziert Artefakte beim Resampling.",
    color: "#176B87",
  },
  {
    number: "02",
    title: "Gaussian Smoothing",
    target: "Rauschen",
    body: "Dämpft hochfrequentes Rauschen und erhält breite spektrale Peaks.",
    color: "#258A8A",
  },
  {
    number: "03",
    title: "sqrt-\nTransformation",
    target: "Dynamik",
    body: "Dämpft dominante Peaks und erhöht den Einfluss schwächerer Signale.",
    color: "#55A878",
  },
  {
    number: "04",
    title: "SNV Normalization",
    target: "Skalierung",
    body: "Entfernt Offset- und Skaleneffekte pro Spektrum.",
    color: "#84A94E",
  },
];

const stepLefts = [64, 270, 476, 682];
for (let i = 0; i < steps.length; i++) {
  const step = steps[i];
  const left = stepLefts[i];
  addRect(`step-accent-${i + 1}`, { left, top: 176, width: 176, height: 6 }, step.color);
  addText(`step-number-${i + 1}`, step.number, { left, top: 198, width: 50, height: 32 }, { fontSize: 22, bold: true, color: step.color });
  addText(`step-target-${i + 1}`, step.target.toUpperCase(), { left: left + 58, top: 204, width: 118, height: 24 }, { fontSize: 12, bold: true, color: "#78838A", alignment: "right" });
  addText(`step-title-${i + 1}`, step.title, { left, top: 246, width: 176, height: 58 }, { fontSize: 21, bold: true, color: "#102B3F" });
  addText(`step-body-${i + 1}`, step.body, { left, top: 320, width: 176, height: 130 }, { fontSize: 16, color: "#384A55" });
}

addRect("rationale-panel", { left: 914, top: 176, width: 302, height: 330 }, "#10384A", true);
addText("rationale-title", "Warum diese Wahl?", { left: 944, top: 207, width: 242, height: 38 }, { fontSize: 22, bold: true, color: "#FFFFFF" });
addText("rationale-main", "Empirische Auswahl", { left: 944, top: 264, width: 242, height: 30 }, { fontSize: 18, bold: true, color: "#A9D7D4" });
addText(
  "rationale-body",
  "Das Screening identifizierte FIR, Gaussian, sqrt und SNV als stärkste Methoden.\n\nDie Kombination prüfen wir über Modelle, Modalitäten und Seeds.\n\nIdentische Eingaben ermöglichen einen fairen Modellvergleich.",
  { left: 944, top: 304, width: 242, height: 184 },
  { fontSize: 15, color: "#FFFFFF" },
);

addRect("talk-track-box", { left: 64, top: 528, width: 1152, height: 132 }, "#E7ECE8", true);
addText("talk-track-label", "KURZANTWORT BEI RÜCKFRAGE", { left: 92, top: 550, width: 290, height: 24 }, { fontSize: 13, bold: true, color: "#2E6F7E" });
addText(
  "talk-track",
  "„Wir haben die Pipeline nicht subjektiv festgelegt. Das Screening zeigte FIR, Gaussian, sqrt und SNV als stärkste Einzelmethoden. Anschliessend prüfen wir ihre Kombination über alle Modelle, Spektren und Seeds.“",
  { left: 92, top: 582, width: 1090, height: 58 },
  { fontSize: 18, bold: true, color: "#173441" },
);

const preview = await deck.export({ slide, format: "png", scale: 1.5 });
await fs.writeFile(path.join(buildDir, "preprocessing-slide-preview.png"), new Uint8Array(await preview.arrayBuffer()));

const draftPath = path.join(buildDir, "preprocessing-slide-draft.pptx");
await (await PresentationFile.exportPptx(deck)).save(draftPath);

const { finalizePresentation } = await import(pathToFileURL(path.join(SKILL_DIR, "container_tools/artifact_tool_utils.mjs")).href);
const stagingDir = path.join(buildDir, ".finalizer");
await fs.mkdir(stagingDir, { recursive: true });
const candidatePath = path.join(stagingDir, "candidate.pptx");
await fs.copyFile(draftPath, candidatePath);

const result = await finalizePresentation({
  workspaceDir,
  candidatePath,
  finalPath: FINAL_PPTX,
  pythonExecutable: RUNTIME_PYTHON,
  integrityValidatorPath: path.join(SKILL_DIR, "container_tools/inspect_presentation_package_integrity.py"),
  layoutValidatorPath: path.join(SKILL_DIR, "container_tools/inspect_presentation_layout_geometry.py"),
  layoutArgs: [
    "--expected-slide-size-emu", "12192000,6858000",
    "--validate-bullet-geometry",
    "--validate-heading-fit",
  ],
  explicitTotalSlideCount: 1,
  requiredNativeTableOwnerSlides: [],
  requiredNativeChartOwnerSlides: [],
  fontPolicy: { basis: "design", families: [fontFamily] },
  verifyArtifactToolImport: true,
  receiptPath: path.join(stagingDir, "preprocessing_selection_slide_final.validation.json"),
});

console.log(JSON.stringify({ final: FINAL_PPTX, preview: path.join(buildDir, "preprocessing-slide-preview.png"), result }, null, 2));
