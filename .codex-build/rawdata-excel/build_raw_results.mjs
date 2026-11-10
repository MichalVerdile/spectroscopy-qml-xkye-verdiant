import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const outputDir = "/Users/leonakryeziu/ZHAW_BSc_Modul/SEM6/BA/spectroscopy-qml-xkye-verdiant/outputs/rawdata-results-seed41-20261104";
const outputPath = `${outputDir}/raw_data_seed41_f1_results.xlsx`;
const previewPresentation = `${outputDir}/preview_presentation.png`;
const previewDetails = `${outputDir}/preview_all_results.png`;

const data = [
  { modality: "¹³C-NMR", key: "cnmr", model: "CNN", micro: 0.582943976357987, macro: 0.09098680012772924, input: 10000, job: "159874" },
  { modality: "¹³C-NMR", key: "cnmr", model: "XGBoost", micro: 0.8493828347739513, macro: 0.5116708145513281, input: 10000, job: "159874" },
  { modality: "¹³C-NMR", key: "cnmr", model: "MPS", micro: 0.7453197340307505, macro: 0.46306909552760356, input: 10000, job: "159874" },
  { modality: "¹³C-NMR", key: "cnmr", model: "TTN", micro: 0.7611661975559526, macro: 0.42078679412744324, input: 10000, job: "160423" },
  { modality: "¹H-NMR", key: "hnmr", model: "CNN", micro: 0.6118567161669543, macro: 0.1111758158925436, input: 10000, job: "159874" },
  { modality: "¹H-NMR", key: "hnmr", model: "XGBoost", micro: 0.8893408275376677, macro: 0.5611636686774136, input: 10000, job: "159874" },
  { modality: "¹H-NMR", key: "hnmr", model: "MPS", micro: 0.7990086997064776, macro: 0.5426641166334035, input: 10000, job: "159874" },
  { modality: "¹H-NMR", key: "hnmr", model: "TTN", micro: 0.7759321959227887, macro: 0.4432466644226846, input: 10000, job: "160423" },
  { modality: "IR", key: "ir", model: "CNN", micro: 0.8797843196465541, macro: 0.633428651496362, input: 1800, job: "159545" },
  { modality: "IR", key: "ir", model: "XGBoost", micro: 0.8639928112922617, macro: 0.5641456971105889, input: 1800, job: "159545" },
  { modality: "IR", key: "ir", model: "MPS", micro: 0.8517030449098747, macro: 0.7021144268238034, input: 1800, job: "159545" },
  { modality: "IR", key: "ir", model: "TTN", micro: 0.8490683980228081, macro: 0.623717461340195, input: 1800, job: "159545" },
  { modality: "MS/MS(-)", key: "msms_neg", model: "CNN", micro: 0.7781833179272941, macro: 0.32997264034203094, input: 10000, job: "159874" },
  { modality: "MS/MS(-)", key: "msms_neg", model: "XGBoost", micro: 0.7886551257225223, macro: 0.38864947051611076, input: 10000, job: "159874" },
  { modality: "MS/MS(-)", key: "msms_neg", model: "MPS", micro: 0.6793509118716633, macro: 0.39278230341783954, input: 10000, job: "159874" },
  { modality: "MS/MS(-)", key: "msms_neg", model: "TTN", micro: 0.7474258523465723, macro: 0.41005401503634387, input: 10000, job: "160423" },
  { modality: "MS/MS(+)", key: "msms_pos", model: "CNN", micro: 0.7420116178291993, macro: 0.24404706896661058, input: 10000, job: "159874" },
  { modality: "MS/MS(+)", key: "msms_pos", model: "XGBoost", micro: 0.7686901251808554, macro: 0.3291107776872146, input: 10000, job: "159874" },
  { modality: "MS/MS(+)", key: "msms_pos", model: "MPS", micro: 0.6760400536297227, macro: 0.3756499641925799, input: 10000, job: "159874" },
  { modality: "MS/MS(+)", key: "msms_pos", model: "TTN", micro: 0.7188405645405029, macro: 0.3590013298776432, input: 10000, job: "160423" },
];

const maxima = new Map();
for (const row of data) {
  const current = maxima.get(row.key) ?? { micro: -Infinity, macro: -Infinity };
  current.micro = Math.max(current.micro, row.micro);
  current.macro = Math.max(current.macro, row.macro);
  maxima.set(row.key, current);
}
for (const row of data) {
  row.bestMicro = row.micro === maxima.get(row.key).micro;
  row.bestMacro = row.macro === maxima.get(row.key).macro;
}

const workbook = Workbook.create();
const presentation = workbook.worksheets.add("Presentation");
const details = workbook.worksheets.add("All Results");
const font = "Times New Roman";
const black = "#000000";
const lightRule = "#B7B7B7";
const darkBlue = "#1F4E78";

presentation.showGridLines = false;
presentation.tabColor = darkBlue;
presentation.getRange("A1:D25").format.font = { name: font, size: 11, color: black };
presentation.mergeCells("A2:D2");
presentation.getRange("A2").values = [["Raw-spectrum functional-group classification performance (Seed 41). Micro- and macro-averaged F1 scores are reported for all models. Best results per modality and metric are highlighted."]];
presentation.getRange("A2:D2").format = {
  font: { name: font, size: 13, bold: true, color: black },
  wrapText: true,
  verticalAlignment: "center",
  horizontalAlignment: "left",
};
presentation.getRange("A4:D4").values = [["Modality", "Model", "Micro-F1", "Macro-F1"]];
presentation.getRange("A4:D4").format = {
  font: { name: font, size: 12, bold: true, color: black },
  horizontalAlignment: "center",
  verticalAlignment: "center",
  borders: {
    top: { style: "medium", color: black },
    bottom: { style: "thin", color: black },
  },
};
presentation.getRange("A5:D24").values = data.map((row) => [row.modality, row.model, row.micro, row.macro]);
presentation.getRange("A5:B24").format.horizontalAlignment = "left";
presentation.getRange("C5:D24").format.horizontalAlignment = "right";
presentation.getRange("C5:D24").format.numberFormat = "0.0000";
presentation.getRange("A5:D24").format.verticalAlignment = "center";
for (const startRow of [5, 9, 13, 17, 21]) {
  presentation.getRange(`A${startRow}:D${startRow}`).format.borders = {
    top: { style: "thin", color: lightRule },
  };
}
presentation.getRange("A24:D24").format.borders = {
  bottom: { style: "medium", color: black },
};
data.forEach((row, index) => {
  const excelRow = index + 5;
  if (row.bestMicro) presentation.getRange(`C${excelRow}`).format.font = { name: font, size: 11, bold: true, color: black };
  if (row.bestMacro) presentation.getRange(`D${excelRow}`).format.font = { name: font, size: 11, bold: true, color: black };
});
presentation.getRange("A1:D25").format.rowHeight = 20;
presentation.getRange("A2:D2").format.rowHeight = 48;
presentation.getRange("A4:D4").format.rowHeight = 24;
presentation.getRange("A:A").format.columnWidth = 18;
presentation.getRange("B:B").format.columnWidth = 17;
presentation.getRange("C:D").format.columnWidth = 15;

details.showGridLines = false;
details.getRange("A1:L28").format.font = { name: "Arial", size: 10, color: "#222222" };
details.mergeCells("A2:L2");
details.getRange("A2").values = [["Raw-spectrum classification results (Seed 41)"]];
details.getRange("A2:L2").format = {
  font: { name: "Arial", size: 14, bold: true, color: "#222222" },
  borders: { bottom: { style: "thin", color: darkBlue } },
  verticalAlignment: "center",
};
details.mergeCells("A3:L3");
details.getRange("A3").values = [["Native input: C-NMR, H-NMR and MS/MS = 10,000 values; IR = 1,800 values. External preprocessing: none. Model-internal normalization remains unchanged."]];
details.getRange("A3:L3").format = {
  font: { name: "Arial", size: 10, italic: true, color: "#555555" },
  wrapText: true,
};
details.mergeCells("A4:L4");
details.getRange("A4").values = [["Training: Seed 41; up to 1,000,000 epochs/estimators; early-stopping patience 15. TTN uses stride 44 only for the 10,000-point native inputs."]];
details.getRange("A4:L4").format = {
  font: { name: "Arial", size: 10, italic: true, color: "#555555" },
  wrapText: true,
};
const headers = ["Modality", "Model", "Seed", "Input length", "Micro-F1", "Macro-F1", "Best Micro-F1", "Best Macro-F1", "External preprocessing", "Training configuration", "Cluster job", "Notes"];
details.getRange("A6:L6").values = [headers];
details.getRange("A6:L6").format = {
  fill: darkBlue,
  font: { name: "Arial", size: 10, bold: true, color: "#FFFFFF" },
  horizontalAlignment: "center",
  verticalAlignment: "center",
  wrapText: true,
  borders: { insideVertical: { style: "thin", color: "#FFFFFF" } },
};
details.getRange("A7:L26").values = data.map((row) => [
  row.modality,
  row.model,
  41,
  row.input,
  row.micro,
  row.macro,
  row.bestMicro ? "Yes" : "No",
  row.bestMacro ? "Yes" : "No",
  "None",
  row.model === "XGBoost" ? "1,000,000 estimators; patience 15" : "1,000,000 epochs; patience 15",
  row.job,
  row.model === "TTN" && row.input === 10000 ? "Window 48; stride 44" : "",
]);
details.getRange("A7:B26").format.horizontalAlignment = "left";
details.getRange("C7:L26").format.verticalAlignment = "center";
details.getRange("C7:F26").format.horizontalAlignment = "right";
details.getRange("G7:I26").format.horizontalAlignment = "center";
details.getRange("J7:L26").format.horizontalAlignment = "left";
details.getRange("C7:C26").format.numberFormat = "0";
details.getRange("D7:D26").format.numberFormat = "#,##0";
details.getRange("E7:F26").format.numberFormat = "0.0000";
for (const startRow of [7, 11, 15, 19, 23]) {
  details.getRange(`A${startRow}:L${startRow}`).format.borders = {
    top: { style: "thin", color: lightRule },
  };
}
details.getRange("A26:L26").format.borders = {
  bottom: { style: "medium", color: darkBlue },
};
data.forEach((row, index) => {
  const excelRow = index + 7;
  if (row.bestMicro) {
    details.getRange(`E${excelRow}`).format.font = { name: "Arial", size: 10, bold: true, color: black };
    details.getRange(`G${excelRow}`).format.font = { name: "Arial", size: 10, bold: true, color: black };
  }
  if (row.bestMacro) {
    details.getRange(`F${excelRow}`).format.font = { name: "Arial", size: 10, bold: true, color: black };
    details.getRange(`H${excelRow}`).format.font = { name: "Arial", size: 10, bold: true, color: black };
  }
});
details.getRange("A1:L28").format.rowHeight = 20;
details.getRange("A2:L2").format.rowHeight = 28;
details.getRange("A3:L4").format.rowHeight = 28;
details.getRange("A6:L6").format.rowHeight = 34;
const widths = [15, 13, 8, 13, 12, 12, 15, 15, 20, 33, 12, 22];
widths.forEach((width, index) => {
  details.getRangeByIndexes(0, index, 28, 1).format.columnWidth = width;
});
details.freezePanes.freezeRows(6);
details.freezePanes.freezeColumns(2);

workbook.recalculate();

const presentationCheck = await workbook.inspect({
  kind: "table",
  range: "Presentation!A2:D24",
  include: "values,formulas",
  tableMaxRows: 30,
  tableMaxCols: 6,
  maxChars: 16000,
});
console.log(presentationCheck.ndjson);
const detailsCheck = await workbook.inspect({
  kind: "table",
  range: "All Results!A2:L26",
  include: "values,formulas",
  tableMaxRows: 30,
  tableMaxCols: 12,
  maxChars: 22000,
});
console.log(detailsCheck.ndjson);
const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
  options: { useRegex: true, maxResults: 300 },
  summary: "final formula error scan",
});
console.log(errors.ndjson);

await fs.mkdir(outputDir, { recursive: true });
const presentationPng = await workbook.render({ sheetName: "Presentation", range: "A1:D25", scale: 2, format: "png" });
await fs.writeFile(previewPresentation, new Uint8Array(await presentationPng.arrayBuffer()));
const detailsPng = await workbook.render({ sheetName: "All Results", range: "A1:L27", scale: 1.2, format: "png" });
await fs.writeFile(previewDetails, new Uint8Array(await detailsPng.arrayBuffer()));

const xlsx = await SpreadsheetFile.exportXlsx(workbook);
await xlsx.save(outputPath);

const saved = await SpreadsheetFile.importXlsx(await FileBlob.load(outputPath));
const savedCheck = await saved.inspect({ kind: "sheet,table", maxChars: 6000, tableMaxRows: 4, tableMaxCols: 12 });
console.log(savedCheck.ndjson);
console.log(JSON.stringify({ outputPath, previewPresentation, previewDetails }));
