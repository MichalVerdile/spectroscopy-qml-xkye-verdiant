import fs from "node:fs/promises";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const outputDir = new URL("./", import.meta.url).pathname;
const results = JSON.parse(await fs.readFile(new URL("./results.json", import.meta.url), "utf8"));
const workbook = Workbook.create();
const font = "Arial";
const navy = "#17365D";
const blue = "#2F75B5";
const lightBlue = "#D9EAF7";
const paleBlue = "#EAF2F8";
const green = "#E2F0D9";
const amber = "#FFF2CC";
const red = "#FCE4D6";
const gray = "#F2F2F2";
const text = "#1F2937";
const configs = [
  "none", "snv", "minmax", "max", "area", "l2", "quantile", "pqn",
  "savgol", "gaussian", "moving_average", "lorentz", "sqrt", "log1p",
  "derivative", "bin_mean", "bin_max", "fir"
];
const models = ["cnn", "xgboost", "mps", "ttn"];
const spectra = ["cnmr", "hnmr", "ir", "msms_pos", "msms_neg"];
const snapshot = "2026-10-26";

function lastCol(index) {
  let n = index + 1;
  let out = "";
  while (n) {
    n--;
    out = String.fromCharCode(65 + (n % 26)) + out;
    n = Math.floor(n / 26);
  }
  return out;
}

function applyBase(sheet) {
  sheet.showGridLines = false;
  const used = sheet.getUsedRange();
  if (used) {
    used.format.font = { name: font, size: 10, color: text };
    used.format.verticalAlignment = "center";
  }
}

function styleTitle(sheet, title, subtitle, endCol = "L") {
  sheet.getRange("A2").values = [[title]];
  sheet.getRange("A2").format.font = { name: font, size: 16, bold: true, color: navy };
  sheet.getRange("A3").values = [[subtitle]];
  sheet.getRange(`A3:${endCol}3`).format.font = { name: font, size: 10, italic: true, color: "#5B6573" };
  sheet.getRange(`A4:${endCol}4`).format.borders = { bottom: { style: "thin", color: "#AAB7C4" } };
}

function styleHeader(range) {
  range.format = {
    fill: navy,
    font: { name: font, size: 10, bold: true, color: "#FFFFFF" },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    wrapText: true,
    borders: { insideVertical: { style: "thin", color: "#FFFFFF" }, bottom: { style: "medium", color: navy } },
  };
}

const raw = workbook.worksheets.add("Alle Resultate");
const rawHeaders = [
  "Preprocessing", "Spektrum", "ML-Modell", "Seed", "Input-Dimension",
  "Train n", "Validation n", "Test n",
  "Val F1 micro", "Val F1 macro", "Val F1 blend", "Val Loss",
  "Test F1 micro", "Test F1 macro", "Test F1 blend", "Test Loss",
  "Test Accuracy", "Test F1 weighted", "Test Precision micro", "Test Precision macro",
  "Test Recall micro", "Test Recall macro", "Best Epoch", "Epochen gelaufen",
  "Trainingszeit (min)", "Quelldatei"
];
const rawKeys = [
  "preprocessing", "spectrum", "model", "seed", "input_dim", "train_size", "val_size", "test_size",
  "val_f1_micro", "val_f1_macro", "val_f1_blend", "val_loss", "test_f1_micro", "test_f1_macro",
  "test_f1_blend", "test_loss", "test_accuracy", "test_f1_weighted", "test_precision_micro",
  "test_precision_macro", "test_recall_micro", "test_recall_macro", "best_epoch", "epochs_completed",
  "fit_minutes", "metric_source"
];
const rawRows = results.map(r => rawKeys.map(k => r[k] ?? null));
raw.getRange(`A1:${lastCol(rawHeaders.length - 1)}${rawRows.length + 1}`).values = [rawHeaders, ...rawRows];
raw.tables.add(`A1:Z${rawRows.length + 1}`, true, "AllResultsTable").style = "TableStyleMedium2";
styleHeader(raw.getRange("A1:Z1"));
raw.freezePanes.freezeRows(1);
raw.freezePanes.freezeColumns(3);
raw.getRange(`I2:V${rawRows.length + 1}`).format.numberFormat = "0.0000";
raw.getRange(`Y2:Y${rawRows.length + 1}`).format.numberFormat = "0.0";
raw.getRange(`I2:K${rawRows.length + 1}`).conditionalFormats.add("colorScale", {
  colors: ["#F8696B", "#FFEB84", "#63BE7B"], thresholds: ["min", { type: "percentile", value: 50 }, "max"]
});
raw.getRange(`M2:O${rawRows.length + 1}`).conditionalFormats.add("colorScale", {
  colors: ["#F8696B", "#FFEB84", "#63BE7B"], thresholds: ["min", { type: "percentile", value: 50 }, "max"]
});
raw.getRange("A:Z").format.autofitColumns();
for (const c of ["A", "B", "C"]) raw.getRange(`${c}:${c}`).format.columnWidth = 15;
raw.getRange("Z:Z").format.columnWidth = 52;
raw.getRange("A1:Z1").format.rowHeight = 32;
applyBase(raw);

const summary = workbook.worksheets.add("Übersicht");
styleTitle(summary, "Preprocessing-Benchmark – alle Resultate", `Stand ${snapshot} · Input 600 · Seed 41 · 18 Parameter × 4 Modelle × 5 Spektren`, "V");
summary.getRange("A6:H6").values = [[
  "Rang", "Preprocessing", "Resultate", "Ø Val F1 micro", "Ø Val F1 macro", "Ø Val F1 blend", "Ø Test F1 micro", "Ø Test F1 macro"
]];
styleHeader(summary.getRange("A6:H6"));
const configMean = name => {
  const values = results.filter(r => r.preprocessing === name).map(r => r.val_f1_blend);
  return values.reduce((sum, value) => sum + value, 0) / values.length;
};
const rankedConfigs = [...configs].sort((a, b) => configMean(b) - configMean(a));
summary.getRange("B7:B24").values = rankedConfigs.map(v => [v]);
const summaryFormulas = rankedConfigs.map((_, i) => {
  const r = 7 + i;
  return [
    `=_xlfn.RANK.EQ(F${r},$F$7:$F$24,0)`, null,
    `=COUNTIF('Alle Resultate'!$A$2:$A$361,B${r})`,
    `=AVERAGEIF('Alle Resultate'!$A$2:$A$361,B${r},'Alle Resultate'!$I$2:$I$361)`,
    `=AVERAGEIF('Alle Resultate'!$A$2:$A$361,B${r},'Alle Resultate'!$J$2:$J$361)`,
    `=AVERAGEIF('Alle Resultate'!$A$2:$A$361,B${r},'Alle Resultate'!$K$2:$K$361)`,
    `=AVERAGEIF('Alle Resultate'!$A$2:$A$361,B${r},'Alle Resultate'!$M$2:$M$361)`,
    `=AVERAGEIF('Alle Resultate'!$A$2:$A$361,B${r},'Alle Resultate'!$N$2:$N$361)`
  ];
});
summary.getRange("A7:H24").formulas = summaryFormulas;
summary.getRange("D7:H24").format.numberFormat = "0.0000";
summary.getRange("A7:A24").format.numberFormat = "0";
summary.getRange("A7:H24").format.borders = { bottom: { style: "thin", color: "#D9E1E8" } };
summary.getRange("A7:H7").format.fill = green;
summary.getRange("F7:F24").conditionalFormats.add("colorScale", {
  colors: ["#F8696B", "#FFEB84", "#63BE7B"], thresholds: ["min", { type: "percentile", value: 50 }, "max"]
});
summary.getRange("A27:D27").values = [["Kernaussage", "Wert", "Definition", "Hinweis"]];
styleHeader(summary.getRange("A27:D27"));
summary.getRange("A28:D31").values = [
  ["Bester Parameter gesamt", rankedConfigs[0], "höchster Ø Val F1 blend", "Blend = (micro + macro) / 2"],
  ["Vollständig ausgewertet", 18, "Preprocessing-Parameter", "je 20 Resultate"],
  ["Resultate im Vergleich", 360, "18 × 4 Modelle × 5 Spektren", "alle geplanten Runs abgeschlossen"],
  ["Fehlende Felder", "leer", "nicht vom jeweiligen Trainer gespeichert", "keine Schätzung/Null-Auffüllung"]
];
summary.getRange("A28:D31").format.borders = { bottom: { style: "thin", color: "#D9E1E8" } };
summary.getRange("B28:B30").format.fill = paleBlue;
summary.getRange("J6:M6").values = [["Preprocessing", "Val micro", "Val macro", "Val blend"]];
styleHeader(summary.getRange("J6:M6"));
summary.getRange("J7:J24").formulas = rankedConfigs.map((_, i) => [[`=B${7 + i}`]]);
summary.getRange("K7:M24").formulas = rankedConfigs.map((_, i) => {
  const r = 7 + i;
  return [`=D${r}`, `=E${r}`, `=F${r}`];
});
summary.getRange("K7:M24").format.numberFormat = "0.0000";
const chart = summary.charts.add("bar", summary.getRange("J6:M24"));
chart.title = "Durchschnittliche Validation-F1-Werte";
chart.titleTextStyle.typeface = font;
chart.legend = { position: "top", textStyle: { typeface: font } };
chart.setPosition("J27", "V47");
if (chart.series.items[0]) chart.series.items[0].fill = "#5B9BD5";
if (chart.series.items[1]) chart.series.items[1].fill = "#ED7D31";
if (chart.series.items[2]) chart.series.items[2].fill = "#70AD47";
summary.getRange("A34:E34").values = [["Spitzenwert", "Wert", "Preprocessing", "ML-Modell", "Spektrum"]];
styleHeader(summary.getRange("A34:E34"));
const bestMetrics = [
  ["Beste Validation F1 micro", "I"],
  ["Beste Validation F1 macro", "J"],
  ["Beste Test F1 micro", "M"],
  ["Beste Test F1 macro", "N"]
];
summary.getRange("A35:A38").values = bestMetrics.map(([label]) => [label]);
summary.getRange("B35:E38").formulas = bestMetrics.map(([, col], i) => {
  const r = 35 + i;
  return [
    `=MAX('Alle Resultate'!$${col}$2:$${col}$361)`,
    `=INDEX('Alle Resultate'!$A$2:$A$361,MATCH(B${r},'Alle Resultate'!$${col}$2:$${col}$361,0))`,
    `=INDEX('Alle Resultate'!$C$2:$C$361,MATCH(B${r},'Alle Resultate'!$${col}$2:$${col}$361,0))`,
    `=INDEX('Alle Resultate'!$B$2:$B$361,MATCH(B${r},'Alle Resultate'!$${col}$2:$${col}$361,0))`
  ];
});
summary.getRange("B35:B38").format.numberFormat = "0.0000";
summary.getRange("A35:E38").format.borders = { bottom: { style: "thin", color: "#D9E1E8" } };
summary.getRange("B35:E38").format.fill = paleBlue;
summary.getRange("A:V").format.autofitColumns();
summary.getRange("A:A").format.columnWidth = 9;
summary.getRange("B:B").format.columnWidth = 16;
summary.getRange("C:H").format.columnWidth = 16;
summary.getRange("A27:A38").format.columnWidth = 27;
summary.getRange("C27:D31").format.columnWidth = 30;
applyBase(summary);

const matrix = workbook.worksheets.add("Modell x Spektrum");
styleTitle(matrix, "Direktvergleich je ML-Modell und Spektrum", "Bestes Preprocessing wird nach Validation F1 blend bestimmt", "V");
function makeMatrix(startRow, metricCol, title) {
  matrix.getRange(`A${startRow}`).values = [[title]];
  matrix.getRange(`A${startRow}`).format.font = { name: font, size: 12, bold: true, color: navy };
  const headerRow = startRow + 2;
  matrix.getRange(`A${headerRow}:V${headerRow}`).values = [["ML-Modell", "Spektrum", ...configs, "Bestwert", "Bester Parameter"]];
  styleHeader(matrix.getRange(`A${headerRow}:V${headerRow}`));
  const pairs = models.flatMap(m => spectra.map(s => [m, s]));
  matrix.getRange(`A${headerRow + 1}:B${headerRow + pairs.length}`).values = pairs;
  const formulas = pairs.map((_, i) => {
    const r = headerRow + 1 + i;
    const avg = configs.map((__, j) => `=AVERAGEIFS('Alle Resultate'!$${metricCol}$2:$${metricCol}$361,'Alle Resultate'!$A$2:$A$361,${lastCol(j + 2)}$${headerRow},'Alle Resultate'!$B$2:$B$361,$B${r},'Alle Resultate'!$C$2:$C$361,$A${r})`);
    return [...avg, `=MAX(C${r}:T${r})`, `=INDEX($C$${headerRow}:$T$${headerRow},1,MATCH(U${r},C${r}:T${r},0))`];
  });
  matrix.getRange(`C${headerRow + 1}:V${headerRow + pairs.length}`).formulas = formulas;
  matrix.getRange(`C${headerRow + 1}:U${headerRow + pairs.length}`).format.numberFormat = "0.0000";
  matrix.getRange(`C${headerRow + 1}:T${headerRow + pairs.length}`).conditionalFormats.add("colorScale", {
    colors: ["#F8696B", "#FFEB84", "#63BE7B"], thresholds: ["min", { type: "percentile", value: 50 }, "max"]
  });
  matrix.getRange(`U${headerRow + 1}:V${headerRow + pairs.length}`).format.fill = lightBlue;
  matrix.getRange(`A${headerRow + 1}:V${headerRow + pairs.length}`).format.borders = { bottom: { style: "thin", color: "#D9E1E8" } };
}
makeMatrix(6, "K", "Validation F1 blend");
makeMatrix(31, "O", "Test F1 blend");
matrix.freezePanes.freezeRows(8);
matrix.freezePanes.freezeColumns(2);
matrix.getRange("A:V").format.autofitColumns();
matrix.getRange("A:B").format.columnWidth = 15;
matrix.getRange("C:U").format.columnWidth = 12;
matrix.getRange("V:V").format.columnWidth = 18;
applyBase(matrix);

const status = workbook.worksheets.add("Run-Status");
styleTitle(status, "Run-Status und Datenabdeckung", `Stand ${snapshot} · alle 18 Parameter vollständig`, "H");
const statusRows = [
  ["none", 20, "vollständig", null, null, "COMPLETED", null],
  ["snv", 20, "vollständig", null, null, "COMPLETED", null],
  ["minmax", 20, "vollständig", null, null, "COMPLETED", null],
  ["max", 20, "vollständig", null, null, "COMPLETED", null],
  ["area", 20, "vollständig", null, null, "COMPLETED", null],
  ["l2", 20, "vollständig", null, null, "COMPLETED", null],
  ["savgol", 20, "vollständig", null, null, "COMPLETED", null],
  ["gaussian", 20, "vollständig", null, null, "COMPLETED", null],
  ["quantile", 20, "vollständig", 155951, "fresko", "COMPLETED", null],
  ["pqn", 20, "vollständig", 155952, "fresko", "COMPLETED", null],
  ["moving_average", 20, "vollständig", 155965, "sanjose", "COMPLETED", null],
  ["lorentz", 20, "vollständig", 156254, "sanjose", "COMPLETED", null],
  ["sqrt", 20, "vollständig", 156259, "sanjose", "COMPLETED", null],
  ["log1p", 20, "vollständig", 156260, "sanjose", "COMPLETED", null],
  ["derivative", 20, "vollständig", 156261, "sanjose", "COMPLETED", null],
  ["bin_mean", 20, "vollständig", 156262, "sanjose", "COMPLETED", null],
  ["bin_max", 20, "vollständig", 156263, "sanjose", "COMPLETED", null],
  ["fir", 20, "vollständig", 156264, "sanjose", "COMPLETED", null]
];
status.getRange("A6:G24").values = [["Preprocessing", "Resultate", "Status", "Job-ID", "Node", "SLURM", "Grund"], ...statusRows];
status.tables.add("A6:G24", true, "RunStatusTable").style = "TableStyleMedium2";
styleHeader(status.getRange("A6:G6"));
status.getRange("A7:G24").format.fill = green;
status.getRange("A:G").format.autofitColumns();
status.getRange("A:A").format.columnWidth = 20;
status.getRange("G:G").format.columnWidth = 24;
applyBase(status);

const notes = workbook.worksheets.add("Methodik");
styleTitle(notes, "Methodik und Quellen", "Wie die Kennzahlen aus den gespeicherten Trainer-Ausgaben gelesen wurden", "F");
notes.getRange("A6:B6").values = [["Thema", "Definition / Quelle"]];
styleHeader(notes.getRange("A6:B6"));
notes.getRange("A7:B16").values = [
  ["Auswertungsumfang", "18 vollständig abgeschlossene Preprocessing-Parameter; je 4 ML-Modelle × 5 Spektren = 20 Resultate, insgesamt 360."],
  ["Validation F1 – CNN/XGBoost", "training_logs.csv, Zeile event=training_finished."],
  ["Validation F1 – MPS", "summary.txt, Abschnitt Best Fold Checkpoint."],
  ["Validation F1 – TTN", "training_log.csv in der in summary.txt angegebenen Best epoch."],
  ["Test F1", "metrics.json; Micro und Macro sind die gespeicherten Abschlusswerte."],
  ["F1 blend", "Arithmetischer Mittelwert aus F1 micro und F1 macro; dient nur zur kompakten Rangierung."],
  ["Loss / Zusatzmetriken", "Nur eingetragen, wenn der jeweilige Trainer sie in CSV/summary.txt gespeichert hat. Leer bedeutet nicht gespeichert, nicht 0."],
  ["Best Epoch – CNN", "Log-Epoche mit höchstem Mittel aus Val F1 micro und Val F1 macro; die angezeigten Val-F1-Abschlusswerte stammen trotzdem aus training_finished."],
  ["Remote-Quelle", "/cluster/home/xkye/spectroscopy-qml-xkye-verdiant/benchmark/preprocessing_training/dim_600/<parameter>/"],
  ["Reproduzierbarkeit", "Input-Dimension 600, Seed 41; Spektren cnmr, hnmr, ir, msms_pos, msms_neg; Modelle cnn, xgboost, mps, ttn."]
];
notes.getRange("A7:B16").format.borders = { bottom: { style: "thin", color: "#D9E1E8" } };
notes.getRange("A7:A16").format.font = { name: font, size: 10, bold: true, color: navy };
notes.getRange("A:A").format.columnWidth = 27;
notes.getRange("B:B").format.columnWidth = 105;
notes.getRange("B7:B16").format.wrapText = true;
notes.getRange("A7:B16").format.rowHeight = 32;
applyBase(notes);

summary.tabColor = navy;
raw.tabColor = blue;
matrix.tabColor = "#5B9BD5";
status.tabColor = "#A5A5A5";
notes.tabColor = "#7F8C8D";

workbook.recalculate();
const checks = [];
checks.push(await workbook.inspect({
  kind: "table", range: "Übersicht!A6:H24", include: "values,formulas", tableMaxRows: 22, tableMaxCols: 10, maxChars: 18000
}));
checks.push(await workbook.inspect({
  kind: "table", range: "Modell x Spektrum!A8:V28", include: "values,formulas", tableMaxRows: 25, tableMaxCols: 22, maxChars: 28000
}));
const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
  options: { useRegex: true, maxResults: 300 },
  summary: "final formula error scan"
});
await fs.writeFile(new URL("./verification.txt", import.meta.url), checks.map(c => c.ndjson).join("\n\n") + "\n\nERROR SCAN\n" + errors.ndjson);

for (const sheetName of ["Übersicht", "Alle Resultate", "Modell x Spektrum", "Run-Status", "Methodik"]) {
  const preview = await workbook.render({ sheetName, autoCrop: "all", scale: sheetName === "Alle Resultate" ? 0.45 : 0.8, format: "png" });
  const safeName = sheetName.normalize("NFD").replace(/[\u0300-\u036f]/g, "").replace(/[^A-Za-z0-9_-]/g, "_");
  await fs.writeFile(`${outputDir}preview_${safeName}.png`, new Uint8Array(await preview.arrayBuffer()));
}

const out = await SpreadsheetFile.exportXlsx(workbook);
await out.save(`${outputDir}preprocessing_ergebnisse_alle_360.xlsx`);
console.log(`Workbook exported to ${outputDir}preprocessing_ergebnisse_alle_360.xlsx`);
