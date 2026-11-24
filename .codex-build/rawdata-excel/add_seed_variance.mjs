import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const inputPath = "/Users/leonakryeziu/Library/CloudStorage/OneDrive-ZHAW/BA/NeurIPs_Paper_Project/seed_f1_results_presentation.xlsx";
const outputDir = "/Users/leonakryeziu/ZHAW_BSc_Modul/SEM6/BA/spectroscopy-qml-xkye-verdiant/outputs/seed-results-with-variance-20261104";
const outputPath = `${outputDir}/seed_f1_results_presentation_with_mean_variance.xlsx`;
const previewPath = `${outputDir}/preview_presentation.png`;

const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(inputPath));
const sheet = workbook.worksheets.getItem("Presentation");
const font = "Times New Roman";
const black = "#000000";

// Extend the existing title across the added summary columns and document the
// statistical definition once, without changing the seed-level result table.
sheet.unmergeCells("A2:L2");
sheet.mergeCells("A2:P2");
sheet.getRange("A2").values = [[
  "Functional-group classification performance by seed and modality. Micro- and macro-averaged F1 scores are shown side by side for all models. Best results within each seed and modality are highlighted. Mean and sample variance (VAR.S, n=5) are calculated across Seeds 41–45.",
]];
sheet.getRange("A2:P2").format = {
  font: { name: font, size: 13, bold: true, color: black },
  wrapText: true,
  horizontalAlignment: "left",
  verticalAlignment: "center",
};

sheet.mergeCells("M4:N4");
sheet.mergeCells("O4:P4");
sheet.getRange("M4").values = [["Micro-F1 summary"]];
sheet.getRange("O4").values = [["Macro-F1 summary"]];
sheet.getRange("M5:P5").values = [["Mean", "Variance", "Mean", "Variance"]];

sheet.getRange("M4:P4").format = {
  font: { name: font, size: 12, bold: true, color: black },
  horizontalAlignment: "center",
  verticalAlignment: "center",
  borders: { top: { style: "medium", color: black } },
};
sheet.getRange("M5:P5").format = {
  font: { name: font, size: 12, bold: true, color: black },
  horizontalAlignment: "center",
  verticalAlignment: "center",
  borders: { bottom: { style: "thin", color: black } },
};

sheet.getRange("M6:P6").formulas = [[
  "=AVERAGE(C6:G6)",
  "=VAR.S(C6:G6)",
  "=AVERAGE(H6:L6)",
  "=VAR.S(H6:L6)",
]];
sheet.getRange("M6:P25").fillDown();
sheet.getRange("M6:P25").format = {
  font: { name: font, size: 11, color: black },
  horizontalAlignment: "right",
  verticalAlignment: "center",
};
sheet.getRange("M6:M25").format.numberFormat = "0.0000";
sheet.getRange("N6:N25").format.numberFormat = "0.000E+00";
sheet.getRange("O6:O25").format.numberFormat = "0.0000";
sheet.getRange("P6:P25").format.numberFormat = "0.000E+00";

for (const row of [6, 10, 14, 18, 22]) {
  sheet.getRange(`M${row}:P${row}`).format.borders = {
    top: { style: "medium", color: black },
  };
}
sheet.getRange("M25:P25").format.borders = {
  bottom: { style: "medium", color: black },
};

sheet.getRange("M:P").format.columnWidth = 15;
sheet.getRange("A2:P2").format.rowHeight = 52;

workbook.recalculate();

const keyCheck = await workbook.inspect({
  kind: "table",
  range: "Presentation!A2:P25",
  include: "values,formulas",
  tableMaxRows: 26,
  tableMaxCols: 16,
  maxChars: 28000,
});
console.log(keyCheck.ndjson);
const formulaCheck = await workbook.inspect({
  kind: "formula",
  sheetId: "Presentation",
  range: "M6:P25",
  maxChars: 12000,
  options: { maxResults: 100 },
});
console.log(formulaCheck.ndjson);
const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
  options: { useRegex: true, maxResults: 300 },
  summary: "final formula error scan",
});
console.log(errors.ndjson);

await fs.mkdir(outputDir, { recursive: true });
const preview = await workbook.render({
  sheetName: "Presentation",
  range: "A1:P25",
  scale: 1.25,
  format: "png",
});
await fs.writeFile(previewPath, new Uint8Array(await preview.arrayBuffer()));

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);

const saved = await SpreadsheetFile.importXlsx(await FileBlob.load(outputPath));
const savedCheck = await saved.inspect({
  kind: "sheet,table,formula",
  sheetId: "Presentation",
  range: "M4:P25",
  maxChars: 12000,
  tableMaxRows: 24,
  tableMaxCols: 4,
});
console.log(savedCheck.ndjson);
console.log(JSON.stringify({ outputPath, previewPath }));
