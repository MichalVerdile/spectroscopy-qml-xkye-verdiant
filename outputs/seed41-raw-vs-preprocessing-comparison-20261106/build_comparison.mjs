import fs from "node:fs/promises";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const outputDir = new URL(".", import.meta.url).pathname;
const outputPath = `${outputDir}seed41_best_model_raw_vs_600.xlsx`;
const previewPath = `${outputDir}preview.png`;

const rows = [
  ["C-NMR", "XGBoost", 0.8493828347739513, 0.5116708145513281, 0.7890503138602921, 0.4131132255237764, 0.8167347577901465, 0.41131202786413235],
  ["H-NMR", "XGBoost", 0.8893408275376677, 0.5611636686774136, 0.7645593016155915, 0.3677340614302773, 0.8099119831221180, 0.38053788722046733],
  ["IR", "CNN", 0.8797843196465541, 0.6334286514963620, 0.8871937706227736, 0.6582987599105201, 0.9048261796306114, 0.7110095282045266],
  ["MS/MS negativ", "XGBoost", 0.7886551257225223, 0.38864947051611076, 0.6265014232031859, 0.11111037161357867, 0.7315782392250079, 0.2638484643095388],
  ["MS/MS positiv", "XGBoost", 0.7686901251808554, 0.3291107776872146, 0.6342628921751166, 0.10474194264308831, 0.7031311906266918, 0.18800336469696927],
];

const workbook = Workbook.create();
const sheet = workbook.worksheets.add("Seed 41 Vergleich");
sheet.showGridLines = false;
sheet.tabColor = "#1F4E78";

sheet.getRange("A2:H2").merge();
sheet.getRange("A2").values = [["Bestes Raw-Modell je Spektrum: Raw vs. 600 Punkte (Seed 41)"]];
sheet.getRange("A2:H2").format = {
  font: { name: "Arial", size: 14, bold: true, color: "#1F1F1F" },
  verticalAlignment: "center",
};
sheet.getRange("A2:H2").format.rowHeight = 26;

sheet.getRange("A3:H3").merge();
sheet.getRange("A3").values = [["Bestes Modell = höchster Micro-F1 im Raw-Lauf. Angezeigt werden Micro- und Macro-F1 desselben Modells in allen drei Eingabevarianten."]];
sheet.getRange("A3:H3").format = {
  font: { name: "Arial", size: 10, italic: true, color: "#595959" },
  verticalAlignment: "center",
};

sheet.getRange("A4:H4").format.borders = {
  bottom: { style: "thin", color: "#7F8C8D" },
};

const headers = [[
  "Spektrum",
  "Bestes Modell (Raw Micro-F1)",
  "Raw/native Micro-F1",
  "Raw/native Macro-F1",
  "600 ohne Preprocessing Micro-F1",
  "600 ohne Preprocessing Macro-F1",
  "600 mit FIR + Gaussian + sqrt + SNV Micro-F1",
  "600 mit FIR + Gaussian + sqrt + SNV Macro-F1",
]];
sheet.getRange("A6:H6").values = headers;
sheet.getRange("A7:H11").values = rows;

const table = sheet.tables.add("A6:H11", true, "Seed41ComparisonTable");
table.style = "TableStyleMedium2";
table.showBandedColumns = false;
table.showFilterButton = true;

sheet.getRange("A6:H6").format = {
  fill: "#1F4E78",
  font: { name: "Arial", size: 10, bold: true, color: "#FFFFFF" },
  horizontalAlignment: "center",
  verticalAlignment: "center",
  wrapText: true,
  borders: {
    insideVertical: { style: "thin", color: "#FFFFFF" },
    bottom: { style: "medium", color: "#17365D" },
  },
};
sheet.getRange("A6:H6").format.rowHeight = 58;

sheet.getRange("A7:H11").format = {
  font: { name: "Arial", size: 10, color: "#1F1F1F" },
  verticalAlignment: "center",
  borders: {
    insideHorizontal: { style: "thin", color: "#D9E2F3" },
  },
};
sheet.getRange("A7:B11").format.horizontalAlignment = "left";
sheet.getRange("C7:H11").format.horizontalAlignment = "right";
sheet.getRange("C7:H11").format.numberFormat = "0.0000";
sheet.getRange("A7:H11").format.rowHeight = 23;

sheet.getRange("C7:D11").format.fill = "#EAF2F8";
sheet.getRange("E7:F11").format.fill = "#F2F2F2";
sheet.getRange("G7:H11").format.fill = "#E9F5EC";
sheet.getRange("B7:B11").format.font = { name: "Arial", size: 10, bold: true, color: "#1F1F1F" };

sheet.getRange("A14:B17").values = [
  ["Auswahlbasis", "Höchster Test-Micro-F1 im Raw-Lauf pro Spektrum, Seed 41."],
  ["Raw/native", "C-NMR, H-NMR und MS/MS: 10.000 Punkte; IR: 1.800 Punkte; kein externes Preprocessing."],
  ["600 ohne PP", "Alle Spektren auf 600 Punkte; externes Preprocessing = none."],
  ["600 mit Kombination", "FIR-Resampling + Gaussian smoothing + sqrt transformation + SNV normalization; 600 Punkte."],
];
sheet.getRange("A14:A17").format = {
  font: { name: "Arial", size: 10, bold: true, color: "#1F4E78" },
  verticalAlignment: "top",
};
sheet.getRange("B14:H17").format = {
  font: { name: "Arial", size: 10, color: "#404040" },
  verticalAlignment: "top",
};
sheet.getRange("B14:B17").format.wrapText = false;

sheet.getRange("A19:B21").values = [
  ["Quellen", "Raw-Ergebnisse: Clusterjobs 159545, 159874 und 160423."],
  ["", "600 ohne Preprocessing: Preprocessing-Ablation, Seed 41, Input-Dimension 600."],
  ["", "600 mit FIR + Gaussian + sqrt + SNV: Clusterjob 157638, Seed 41."],
];
sheet.getRange("A19:A21").format = {
  font: { name: "Arial", size: 9, bold: true, color: "#595959" },
  verticalAlignment: "top",
};
sheet.getRange("B19:H21").format = {
  font: { name: "Arial", size: 9, italic: true, color: "#666666" },
  verticalAlignment: "top",
};

sheet.getRange("A1:H21").format.verticalAlignment = "center";
sheet.getRange("A:A").format.columnWidth = 17;
sheet.getRange("B:B").format.columnWidth = 27;
sheet.getRange("C:D").format.columnWidth = 18;
sheet.getRange("E:F").format.columnWidth = 22;
sheet.getRange("G:H").format.columnWidth = 29;

workbook.recalculate();

const tableCheck = await workbook.inspect({
  kind: "table",
  range: "'Seed 41 Vergleich'!A2:H21",
  include: "values,formulas",
  tableMaxRows: 24,
  tableMaxCols: 10,
});
console.log("TABLE_CHECK");
console.log(tableCheck.ndjson);

const errorCheck = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
  options: { useRegex: true, maxResults: 100 },
  summary: "final formula error scan",
});
console.log("ERROR_CHECK");
console.log(errorCheck.ndjson);

const preview = await workbook.render({
  sheetName: "Seed 41 Vergleich",
  range: "A1:H21",
  scale: 1.5,
  format: "png",
});
await fs.writeFile(previewPath, new Uint8Array(await preview.arrayBuffer()));

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);
console.log(`OUTPUT=${outputPath}`);
console.log(`PREVIEW=${previewPath}`);
