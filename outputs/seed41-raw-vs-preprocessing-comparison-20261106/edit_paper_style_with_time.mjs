import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const outputDir = new URL(".", import.meta.url).pathname;
const inputPath = `${outputDir}seed41_best_model_raw_vs_600.xlsx`;
const outputPath = `${outputDir}seed41_best_model_raw_vs_600_paper_style_with_time.xlsx`;
const previewPath = `${outputDir}preview_paper_style_with_time.png`;

const source = await FileBlob.load(inputPath);
const workbook = await SpreadsheetFile.importXlsx(source);
const sheet = workbook.worksheets.getItem("Seed 41 Vergleich");

for (const table of sheet.tables.items) table.delete();
sheet.unmergeCells("A2:H2");
sheet.unmergeCells("A3:H3");
sheet.getRange("A1:K30").clear({ applyTo: "all" });
sheet.showGridLines = false;
sheet.tabColor = "#000000";

const rows = [
  ["¹³C-NMR", "XGBoost", 0.8493828347739513, 0.5116708145513281, 8563.559290691977 / 60, 0.7890503138602921, 0.4131132255237764, 3.986184904233475, 0.8167347577901465, 0.41131202786413235, 855.5450218319893 / 60],
  ["¹H-NMR", "XGBoost", 0.8893408275376677, 0.5611636686774136, 15780.42388216802 / 60, 0.7645593016155915, 0.3677340614302773, 4.027280925299662, 0.8099119831221180, 0.38053788722046733, 1262.8221524509136 / 60],
  ["IR", "CNN", 0.8797843196465541, 0.6334286514963620, 1897.4462864629459 / 60, 0.8871937706227736, 0.6582987599105201, 39.871537267864916, 0.9048261796306114, 0.7110095282045266, 1839.4564971120562 / 60],
  ["MS/MS(-)", "XGBoost", 0.7886551257225223, 0.38864947051611076, 6746.768216330092 / 60, 0.6265014232031859, 0.11111037161357867, 2.499847331784743, 0.7315782392250079, 0.2638484643095388, 879.6951281179208 / 60],
  ["MS/MS(+)", "XGBoost", 0.7686901251808554, 0.3291107776872146, 8454.306766472058 / 60, 0.6342628921751166, 0.10474194264308831, 2.5515991162663947, 0.7031311906266918, 0.18800336469696927, 976.803805698175 / 60],
];

sheet.mergeCells("A1:A2");
sheet.mergeCells("B1:B2");
sheet.mergeCells("C1:E1");
sheet.mergeCells("F1:H1");
sheet.mergeCells("I1:K1");

sheet.getRange("A1").values = [["Spektrum"]];
sheet.getRange("B1").values = [["Bestes Modell"]];
sheet.getRange("C1").values = [["Raw / native"]];
sheet.getRange("F1").values = [["600 ohne Preprocessing"]];
sheet.getRange("I1").values = [["600 mit FIR + Gaussian + sqrt + SNV"]];

sheet.getRange("C2:K2").values = [[
  "Micro-F1", "Macro-F1", "Trainingszeit (min)",
  "Micro-F1", "Macro-F1", "Trainingszeit (min)",
  "Micro-F1", "Macro-F1", "Trainingszeit (min)",
]];
sheet.getRange("A3:K7").values = rows;

sheet.getRange("A1:K14").format.font = {
  name: "Times New Roman",
  size: 11,
  color: "#000000",
};
sheet.getRange("A1:K2").format.font = {
  name: "Times New Roman",
  size: 11,
  bold: true,
  color: "#000000",
};
sheet.getRange("A1:K2").format.horizontalAlignment = "center";
sheet.getRange("A1:K2").format.verticalAlignment = "center";
sheet.getRange("A1:K2").format.wrapText = true;
sheet.getRange("A1:K1").format.rowHeight = 25;
sheet.getRange("A2:K2").format.rowHeight = 36;

sheet.getRange("A3:B7").format.horizontalAlignment = "left";
sheet.getRange("C3:K7").format.horizontalAlignment = "right";
sheet.getRange("A3:K7").format.verticalAlignment = "center";
sheet.getRange("A3:K7").format.rowHeight = 24;
sheet.getRange("B3:B7").format.font = {
  name: "Times New Roman",
  size: 11,
  bold: true,
  color: "#000000",
};

sheet.getRange("C3:D7").format.numberFormat = "0.0000";
sheet.getRange("F3:G7").format.numberFormat = "0.0000";
sheet.getRange("I3:J7").format.numberFormat = "0.0000";
sheet.getRange("E3:E7").format.numberFormat = "0.0";
sheet.getRange("H3:H7").format.numberFormat = "0.0";
sheet.getRange("K3:K7").format.numberFormat = "0.0";

sheet.getRange("A1:K1").format.borders = {
  top: { style: "medium", color: "#000000" },
};
sheet.getRange("A2:K2").format.borders = {
  bottom: { style: "thin", color: "#000000" },
};
for (const row of [3, 4, 5, 6]) {
  sheet.getRange(`A${row}:K${row}`).format.borders = {
    bottom: { style: "thin", color: "#A6A6A6" },
  };
}
sheet.getRange("A7:K7").format.borders = {
  bottom: { style: "medium", color: "#000000" },
};
for (const col of ["B", "E", "H"]) {
  sheet.getRange(`${col}1:${col}7`).format.borders = {
    right: { style: "thin", color: "#7F7F7F" },
  };
}

const boldCells = [
  "C3", "D3", "H3",
  "C4", "D4", "H4",
  "I5", "J5", "K5",
  "C6", "D6", "H6",
  "C7", "D7", "H7",
];
for (const address of boldCells) {
  sheet.getRange(address).format.font = {
    name: "Times New Roman",
    size: 11,
    bold: true,
    color: "#000000",
  };
}

sheet.getRange("A9:K9").merge();
sheet.getRange("A10:K10").merge();
sheet.getRange("A11:K11").merge();
sheet.getRange("A13:K13").merge();
sheet.getRange("A9").values = [["Auswahl: Pro Spektrum wird das Modell mit dem höchsten Raw-Test-Micro-F1 gezeigt. Fett markiert ist innerhalb einer Zeile der beste F1-Wert beziehungsweise die kürzeste Trainingszeit."]];
sheet.getRange("A10").values = [["Trainingszeit: reine Fit-Zeit des Modells in Minuten; Datenladen, Preprocessing/Cache-Erstellung und Testvorhersage sind nicht enthalten."]];
sheet.getRange("A11").values = [["Eingabe: Raw/native = 10.000 Punkte für NMR und MS/MS, 1.800 Punkte für IR; die beiden anderen Varianten verwenden 600 Punkte. Alle Werte: Seed 41."]];
sheet.getRange("A13").values = [["Quellen: Raw-Trainingslogs (Clusterjobs 159545/159874/160423), 600-Ablation (none) und kombinierte Pipeline (Clusterjob 157638)."]];
sheet.getRange("A9:K13").format.font = {
  name: "Times New Roman",
  size: 9,
  italic: true,
  color: "#404040",
};
sheet.getRange("A9:K13").format.horizontalAlignment = "left";
sheet.getRange("A9:K13").format.verticalAlignment = "center";
sheet.getRange("A9:K11").format.rowHeight = 20;
sheet.getRange("A13:K13").format.rowHeight = 20;

sheet.getRange("A:A").format.columnWidth = 16;
sheet.getRange("B:B").format.columnWidth = 18;
for (const col of ["C", "D", "F", "G", "I", "J"]) {
  sheet.getRange(`${col}:${col}`).format.columnWidth = 13;
}
for (const col of ["E", "H", "K"]) {
  sheet.getRange(`${col}:${col}`).format.columnWidth = 18;
}

workbook.recalculate();

const tableCheck = await workbook.inspect({
  kind: "table",
  range: "'Seed 41 Vergleich'!A1:K13",
  include: "values,formulas",
  tableMaxRows: 16,
  tableMaxCols: 12,
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
  range: "A1:K13",
  scale: 2,
  format: "png",
});
await fs.writeFile(previewPath, new Uint8Array(await preview.arrayBuffer()));

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);
console.log(`OUTPUT=${outputPath}`);
console.log(`PREVIEW=${previewPath}`);
