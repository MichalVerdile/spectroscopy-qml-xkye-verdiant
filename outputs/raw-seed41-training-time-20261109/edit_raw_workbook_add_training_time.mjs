import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const sourcePath = "/Users/leonakryeziu/Library/CloudStorage/OneDrive-ZHAW/BA/NeurIPs_Paper_Project/raw_data_seed41_f1_results.xlsx";
const outputDir = "/Users/leonakryeziu/ZHAW_BSc_Modul/SEM6/BA/spectroscopy-qml-xkye-verdiant/outputs/raw-seed41-training-time-20261109";
const outputPath = `${outputDir}/raw_data_seed41_f1_results_with_training_time.xlsx`;
const timesPath = `${outputDir}/raw_seed41_training_times.tsv`;

const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(sourcePath));
const presentation = workbook.worksheets.getItem("Presentation");
const details = workbook.worksheets.getItem("All Results");

const originalPresentation = JSON.stringify(presentation.getRange("A1:D24").values);
const originalDetails = JSON.stringify(details.getRange("A1:L26").values);

const timeLines = (await fs.readFile(timesPath, "utf8")).trim().split("\n").slice(1);
const times = new Map(
  timeLines.map((line) => {
    const [modality, model, minutes] = line.split("\t");
    return [`${modality}|${model}`, Number(minutes)];
  }),
);

function valuesForRows(sheet, startRow, endRow) {
  const rows = sheet.getRange(`A${startRow}:B${endRow}`).values;
  return rows.map(([modality, model]) => {
    const key = `${modality}|${model}`;
    if (!times.has(key)) throw new Error(`Missing training time for ${key}`);
    return [times.get(key)];
  });
}

presentation.getRange("E1:E25").copyFrom(presentation.getRange("D1:D25"), "all");
presentation.getRange("E1:E25").clear({ applyTo: "contents" });
presentation.getRange("E4").values = [["Training time (min)"]];
presentation.getRange("E5:E24").values = valuesForRows(presentation, 5, 24);
presentation.getRange("E5:E24").format.font = {
  name: "Times New Roman",
  size: 11,
  bold: false,
  color: "#000000",
};
presentation.getRange("E5:E24").format.horizontalAlignment = "right";
presentation.getRange("E5:E24").format.numberFormat = "0.0";
presentation.getRange("E4").format.wrapText = true;
presentation.getRange("E1:E25").format.columnWidth = 20;
presentation.getRange("A2").values = [[
  "Raw-spectrum functional-group classification performance (Seed 41). Micro- and macro-averaged F1 scores and pure model-fit times are reported for all models. Best F1 results per modality and metric are highlighted.",
]];

details.getRange("M1:M28").copyFrom(details.getRange("L1:L28"), "all");
details.getRange("M1:M28").clear({ applyTo: "contents" });
details.getRange("M6").values = [["Training time (min)"]];
details.getRange("M7:M26").values = valuesForRows(details, 7, 26);
details.getRange("M7:M26").format.numberFormat = "0.0";
details.getRange("M7:M26").format.horizontalAlignment = "right";
details.getRange("M6").format.wrapText = true;
details.getRange("M1:M28").format.columnWidth = 18;
details.getRange("A4").values = [[
  "Training: Seed 41; up to 1,000,000 epochs/estimators; early-stopping patience 15. TTN uses stride 44 only for the 10,000-point native inputs. Training time is pure model-fit time in minutes.",
]];

const currentPresentation = presentation.getRange("A1:D24").values;
const expectedPresentation = JSON.parse(originalPresentation);
expectedPresentation[1][0] = "Raw-spectrum functional-group classification performance (Seed 41). Micro- and macro-averaged F1 scores and pure model-fit times are reported for all models. Best F1 results per modality and metric are highlighted.";
if (JSON.stringify(currentPresentation) !== JSON.stringify(expectedPresentation)) {
  throw new Error("Unexpected change in original Presentation cells");
}

const currentDetails = details.getRange("A1:L26").values;
const expectedDetails = JSON.parse(originalDetails);
expectedDetails[3][0] = "Training: Seed 41; up to 1,000,000 epochs/estimators; early-stopping patience 15. TTN uses stride 44 only for the 10,000-point native inputs. Training time is pure model-fit time in minutes.";
if (JSON.stringify(currentDetails) !== JSON.stringify(expectedDetails)) {
  throw new Error("Unexpected change in original All Results cells");
}

workbook.recalculate();

const presentationCheck = await workbook.inspect({
  kind: "table",
  range: "Presentation!A2:E24",
  include: "values,formulas",
  tableMaxRows: 30,
  tableMaxCols: 6,
});
console.log("PRESENTATION_CHECK");
console.log(presentationCheck.ndjson);

const detailCheck = await workbook.inspect({
  kind: "table",
  range: "'All Results'!A2:M26",
  include: "values,formulas",
  tableMaxRows: 30,
  tableMaxCols: 14,
});
console.log("DETAIL_CHECK");
console.log(detailCheck.ndjson);

const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
  options: { useRegex: true, maxResults: 300 },
  summary: "final formula error scan",
});
console.log("ERROR_CHECK");
console.log(errors.ndjson);

for (const [sheetName, range, fileName] of [
  ["Presentation", "A1:E25", "preview_Presentation_with_training_time.png"],
  ["All Results", "A1:M28", "preview_All_Results_with_training_time.png"],
]) {
  const rendered = await workbook.render({ sheetName, range, scale: 2, format: "png" });
  await fs.writeFile(`${outputDir}/${fileName}`, new Uint8Array(await rendered.arrayBuffer()));
}

await fs.mkdir(outputDir, { recursive: true });
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);
console.log(`OUTPUT=${outputPath}`);
