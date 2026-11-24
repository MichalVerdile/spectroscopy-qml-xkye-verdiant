import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const inputPath = "/Users/leonakryeziu/Library/CloudStorage/OneDrive-ZHAW/BA/NeurIPs_Paper_Project/seed_f1_results_presentation_with_mean_variance.xlsx";
const outputPath = new URL("./seed_f1_results_presentation_with_mean_variance_and_training_time.xlsx", import.meta.url).pathname;
const timesPath = new URL("./training_times.tsv", import.meta.url).pathname;

const spectrumKey = {
  "C-NMR": "cnmr",
  "H-NMR": "hnmr",
  "IR": "ir",
  "MS/MS negativ": "msms_neg",
  "MS/MS positiv": "msms_pos",
};

const modelKey = {
  "CNN": "cnn",
  "XGBoost": "xgboost",
  "MPS": "mps",
  "TTN": "ttn",
};

const presentationSpectrumKey = {
  "¹³C-NMR": "cnmr",
  "¹H-NMR": "hnmr",
  "IR": "ir",
  "MS/MS(-)": "msms_neg",
  "MS/MS(+)": "msms_pos",
};

const timeText = await fs.readFile(timesPath, "utf8");
const timeMap = new Map();
for (const line of timeText.trim().split(/\r?\n/).slice(1)) {
  const [spectrum, model, seedText, minutesText] = line.split("\t");
  const minutes = Number(minutesText);
  if (!Number.isFinite(minutes)) throw new Error(`Invalid training time: ${line}`);
  timeMap.set(`${spectrum}|${model}|${seedText}`, minutes);
}
if (timeMap.size !== 100) throw new Error(`Expected 100 training times, found ${timeMap.size}`);

const input = await FileBlob.load(inputPath);
const workbook = await SpreadsheetFile.importXlsx(input);
const presentation = workbook.worksheets.getItem("Presentation");
const allResults = workbook.worksheets.getItem("All Results");

const originalRows = allResults.getRange("A7:H106").values;
const presentationLabels = presentation.getRange("A6:B25").values;
if (originalRows.length !== 100) throw new Error(`Expected 100 result rows, found ${originalRows.length}`);

const expandedRows = originalRows.map((row, index) => {
  const [spectrum, model, seed, micro, macro, comparability, trainingConfig, source] = row;
  const sKey = spectrumKey[String(spectrum)];
  const mKey = modelKey[String(model)];
  const key = `${sKey}|${mKey}|${seed}`;
  const minutes = timeMap.get(key);
  if (!sKey || !mKey || !Number.isFinite(minutes)) {
    throw new Error(`Missing training time at result row ${index + 7}: ${key}`);
  }
  return [spectrum, model, seed, micro, macro, minutes, comparability, trainingConfig, source];
});

const resultLookup = new Map();
expandedRows.forEach((row, index) => {
  const key = `${spectrumKey[String(row[0])]}|${modelKey[String(row[1])]}|${row[2]}`;
  resultLookup.set(key, { row, detailRow: index + 7 });
});
if (resultLookup.size !== 100) throw new Error(`Expected 100 unique result rows, found ${resultLookup.size}`);

// Rebuild the detail sheet with one added numeric time column.
for (const table of allResults.tables.items) table.delete();
allResults.getRange("A1:I106").clear({ applyTo: "all" });
allResults.showGridLines = false;

allResults.getRange("A2").values = [["Alle F1-Ergebnisse und Trainingszeiten nach Seed, Spektrum und Modell"]];
allResults.getRange("A3").values = [["Preprocessing: FIR + Gaussian + sqrt + SNV | Input-Dimension: 600 | Trainingszeit: reine Modell-Fit-Zeit in Minuten"]];
allResults.getRange("A4").values = [["Quelle: Clusterjob 157638 (Seeds 41, 43, 44, 45), Clusterjob 158383 (Seed 42) und zugehörige Trainingslogs/Summaries"]];
allResults.getRange("A6:I6").values = [[
  "Spektrum", "Modell", "Seed", "F1 Micro", "F1 Macro", "Trainingszeit (min)",
  "Vergleichbarkeit", "Trainingskonfiguration", "Quelle",
]];
allResults.getRange("A7:I106").values = expandedRows;

const detailTable = allResults.tables.add("A6:I106", true, "AllResultsWithTimeTable");
detailTable.style = "TableStyleMedium2";
detailTable.showBandedColumns = false;
detailTable.showFilterButton = true;

allResults.getRange("A2:I2").format.font = { name: "Arial", size: 15, bold: true, color: "#1F1F1F" };
allResults.getRange("A3:I3").format.font = { name: "Arial", size: 10, italic: true, color: "#595959" };
allResults.getRange("A4:I4").format.font = { name: "Arial", size: 10, color: "#595959" };
allResults.getRange("A6:I6").format = {
  fill: "#1F4E78",
  font: { name: "Arial", size: 10, bold: true, color: "#FFFFFF" },
  horizontalAlignment: "center",
  verticalAlignment: "center",
};
allResults.getRange("A7:I106").format.font = { name: "Arial", size: 10, color: "#000000" };
allResults.getRange("A7:B106").format.horizontalAlignment = "left";
allResults.getRange("C7:F106").format.horizontalAlignment = "right";
allResults.getRange("G7:I106").format.horizontalAlignment = "left";
allResults.getRange("C7:C106").format.numberFormat = "0";
allResults.getRange("D7:E106").format.numberFormat = "0.0000";
allResults.getRange("F7:F106").format.numberFormat = "0.0";

for (let i = 0; i < expandedRows.length; i++) {
  const rowNumber = i + 7;
  const comparability = String(expandedRows[i][6] ?? "");
  const seed = Number(expandedRows[i][2]);
  if (seed === 42 || comparability !== "Comparable") {
    allResults.getRange(`A${rowNumber}:I${rowNumber}`).format.fill = "#FFF2CC";
    allResults.getRange(`G${rowNumber}`).format.font = { name: "Arial", size: 10, bold: true, color: "#9C6500" };
  }
}

allResults.getRange("A:A").format.columnWidth = 18;
allResults.getRange("B:B").format.columnWidth = 14;
allResults.getRange("C:C").format.columnWidth = 9;
allResults.getRange("D:E").format.columnWidth = 12;
allResults.getRange("F:F").format.columnWidth = 20;
allResults.getRange("G:G").format.columnWidth = 17;
allResults.getRange("H:H").format.columnWidth = 31;
allResults.getRange("I:I").format.columnWidth = 21;
allResults.freezePanes.freezeRows(6);

// Rebuild the paper-style presentation with training time per seed and summary.
for (const range of ["A2:P2", "A4:A5", "B4:B5", "C4:G4", "H4:L4", "M4:N4", "O4:P4"]) {
  try { presentation.unmergeCells(range); } catch {}
}
presentation.getRange("A1:W107").clear({ applyTo: "all" });
presentation.showGridLines = false;

presentation.mergeCells("A2:W2");
presentation.getRange("A2").values = [["Functional-group classification performance and training time by seed and modality. Micro-F1, Macro-F1 and model fit time are reported for all models. Best scores and shortest fit times within each seed and modality are highlighted. Mean and sample variance (VAR.S, n=5) are calculated across Seeds 41–45."]];

presentation.mergeCells("A4:A5");
presentation.mergeCells("B4:B5");
presentation.mergeCells("C4:G4");
presentation.mergeCells("H4:L4");
presentation.mergeCells("M4:Q4");
presentation.mergeCells("R4:S4");
presentation.mergeCells("T4:U4");
presentation.mergeCells("V4:W4");

presentation.getRange("A4").values = [["Modality"]];
presentation.getRange("B4").values = [["Model"]];
presentation.getRange("C4").values = [["Micro-F1"]];
presentation.getRange("H4").values = [["Macro-F1"]];
presentation.getRange("M4").values = [["Training time (min)"]];
presentation.getRange("R4").values = [["Micro-F1 summary"]];
presentation.getRange("T4").values = [["Macro-F1 summary"]];
presentation.getRange("V4").values = [["Training-time summary"]];
presentation.getRange("C5:W5").values = [[
  "Seed 41", "Seed 42", "Seed 43", "Seed 44", "Seed 45",
  "Seed 41", "Seed 42", "Seed 43", "Seed 44", "Seed 45",
  "Seed 41", "Seed 42", "Seed 43", "Seed 44", "Seed 45",
  "Mean", "Variance", "Mean", "Variance", "Mean", "Variance",
]];
presentation.getRange("A6:B25").values = presentationLabels;

for (let p = 0; p < 20; p++) {
  const pRow = p + 6;
  const pSpectrum = presentationSpectrumKey[String(presentationLabels[p][0])];
  const pModel = modelKey[String(presentationLabels[p][1])];
  if (!pSpectrum || !pModel) throw new Error(`Unknown presentation label at row ${pRow}`);
  for (let seedIndex = 0; seedIndex < 5; seedIndex++) {
    const seed = 41 + seedIndex;
    const match = resultLookup.get(`${pSpectrum}|${pModel}|${seed}`);
    if (!match) throw new Error(`Missing presentation source: ${pSpectrum}|${pModel}|${seed}`);
    const detailRow = match.detailRow;
    presentation.getCell(pRow - 1, 2 + seedIndex).formulas = [[`='All Results'!D${detailRow}`]];
    presentation.getCell(pRow - 1, 7 + seedIndex).formulas = [[`='All Results'!E${detailRow}`]];
    presentation.getCell(pRow - 1, 12 + seedIndex).formulas = [[`='All Results'!F${detailRow}`]];
  }
  presentation.getRange(`R${pRow}`).formulas = [[`=AVERAGE(C${pRow}:G${pRow})`]];
  presentation.getRange(`S${pRow}`).formulas = [[`=_xlfn.VAR.S(C${pRow}:G${pRow})`]];
  presentation.getRange(`T${pRow}`).formulas = [[`=AVERAGE(H${pRow}:L${pRow})`]];
  presentation.getRange(`U${pRow}`).formulas = [[`=_xlfn.VAR.S(H${pRow}:L${pRow})`]];
  presentation.getRange(`V${pRow}`).formulas = [[`=AVERAGE(M${pRow}:Q${pRow})`]];
  presentation.getRange(`W${pRow}`).formulas = [[`=_xlfn.VAR.S(M${pRow}:Q${pRow})`]];
}

presentation.getRange("A2:W25").format.font = { name: "Times New Roman", size: 11, color: "#000000" };
presentation.getRange("A2:W2").format = {
  font: { name: "Times New Roman", size: 13, bold: true, color: "#000000" },
  horizontalAlignment: "left",
  verticalAlignment: "center",
  wrapText: true,
};
presentation.getRange("A2:W2").format.rowHeight = 34;
presentation.getRange("A4:W5").format.font = { name: "Times New Roman", size: 12, bold: true, color: "#000000" };
presentation.getRange("A4:W5").format.horizontalAlignment = "center";
presentation.getRange("A4:W5").format.verticalAlignment = "center";
presentation.getRange("A4:W4").format.rowHeight = 24;
presentation.getRange("A5:W5").format.rowHeight = 24;
presentation.getRange("A4:W4").format.borders = { top: { style: "medium", color: "#000000" } };
presentation.getRange("A5:W5").format.borders = { bottom: { style: "thin", color: "#000000" } };

presentation.getRange("A6:B25").format.horizontalAlignment = "left";
presentation.getRange("C6:W25").format.horizontalAlignment = "right";
presentation.getRange("A6:W25").format.verticalAlignment = "center";
presentation.getRange("A6:W25").format.rowHeight = 21;
presentation.getRange("C6:L25").format.numberFormat = "0.0000";
presentation.getRange("M6:Q25").format.numberFormat = "0.0";
presentation.getRange("R6:R25").format.numberFormat = "0.0000";
presentation.getRange("S6:S25").format.numberFormat = "0.000E+00";
presentation.getRange("T6:T25").format.numberFormat = "0.0000";
presentation.getRange("U6:U25").format.numberFormat = "0.000E+00";
presentation.getRange("V6:V25").format.numberFormat = "0.0";
presentation.getRange("W6:W25").format.numberFormat = "0.00";

for (const startRow of [6, 10, 14, 18, 22]) {
  presentation.getRange(`A${startRow}:W${startRow}`).format.borders = { top: { style: "thin", color: "#000000" } };
}
presentation.getRange("A25:W25").format.borders = { bottom: { style: "medium", color: "#000000" } };
for (const col of ["B", "G", "L", "Q", "S", "U"]) {
  presentation.getRange(`${col}4:${col}25`).format.borders = { right: { style: "thin", color: "#7F7F7F" } };
}

// Highlight per-seed best Micro/Macro F1 and shortest fit time within each modality.
for (let modality = 0; modality < 5; modality++) {
  const pSpectrum = presentationSpectrumKey[String(presentationLabels[modality * 4][0])];
  for (let seedIndex = 0; seedIndex < 5; seedIndex++) {
    const seed = 41 + seedIndex;
    const modalityRows = [];
    for (let model = 0; model < 4; model++) {
      const pModel = modelKey[String(presentationLabels[modality * 4 + model][1])];
      const match = resultLookup.get(`${pSpectrum}|${pModel}|${seed}`);
      if (!match) throw new Error(`Missing highlight source: ${pSpectrum}|${pModel}|${seed}`);
      modalityRows.push(match.row);
    }
    const bestMicro = Math.max(...modalityRows.map(row => Number(row[3])));
    const bestMacro = Math.max(...modalityRows.map(row => Number(row[4])));
    const bestTime = Math.min(...modalityRows.map(row => Number(row[5])));
    for (let model = 0; model < 4; model++) {
      const result = modalityRows[model];
      const pRow = 6 + modality * 4 + model;
      for (const [colIndex, value, best] of [
        [2 + seedIndex, Number(result[3]), bestMicro],
        [7 + seedIndex, Number(result[4]), bestMacro],
        [12 + seedIndex, Number(result[5]), bestTime],
      ]) {
        if (Math.abs(value - best) < 1e-12) {
          presentation.getCell(pRow - 1, colIndex).format.font = { name: "Times New Roman", size: 11, bold: true, color: "#000000" };
        }
      }
    }
  }
}

presentation.getRange("A:A").format.columnWidth = 15;
presentation.getRange("B:B").format.columnWidth = 13;
for (const col of ["C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]) {
  presentation.getRange(`${col}:${col}`).format.columnWidth = 11;
}
for (const col of ["M", "N", "O", "P", "Q"]) {
  presentation.getRange(`${col}:${col}`).format.columnWidth = 12;
}
for (const col of ["R", "S", "T", "U", "V", "W"]) {
  presentation.getRange(`${col}:${col}`).format.columnWidth = 13;
}

workbook.recalculate();

// Verify source F1 values were preserved and every time was inserted.
const rebuiltRows = allResults.getRange("A7:I106").values;
for (let i = 0; i < originalRows.length; i++) {
  const oldRow = originalRows[i];
  const newRow = rebuiltRows[i];
  for (const [oldIndex, newIndex] of [[0,0], [1,1], [2,2], [3,3], [4,4], [5,6], [6,7], [7,8]]) {
    if (String(oldRow[oldIndex] ?? "") !== String(newRow[newIndex] ?? "")) {
      throw new Error(`Value preservation failed at row ${i + 7}, columns ${oldIndex}/${newIndex}`);
    }
  }
  if (!Number.isFinite(Number(newRow[5]))) throw new Error(`Missing time at row ${i + 7}`);
}

const presentationCheck = await workbook.inspect({
  kind: "table",
  range: "Presentation!A2:W25",
  include: "values,formulas",
  tableMaxRows: 26,
  tableMaxCols: 24,
});
console.log("PRESENTATION_CHECK");
console.log(presentationCheck.ndjson);

const detailCheck = await workbook.inspect({
  kind: "table",
  range: "'All Results'!A2:I12",
  include: "values,formulas",
  tableMaxRows: 14,
  tableMaxCols: 10,
});
console.log("DETAIL_CHECK");
console.log(detailCheck.ndjson);

const errorCheck = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
  options: { useRegex: true, maxResults: 300 },
  summary: "final formula error scan",
});
console.log("ERROR_CHECK");
console.log(errorCheck.ndjson);

for (const sheetName of ["Presentation", "All Results"]) {
  const preview = await workbook.render({ sheetName, autoCrop: "all", scale: 1, format: "png" });
  const safe = sheetName.replace(/\s+/g, "_");
  await fs.writeFile(new URL(`./preview_${safe}_with_time.png`, import.meta.url), new Uint8Array(await preview.arrayBuffer()));
}

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);
console.log(`OUTPUT=${outputPath}`);
