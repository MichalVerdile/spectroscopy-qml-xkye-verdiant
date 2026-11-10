import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const inputPath = "/Users/leonakryeziu/Library/CloudStorage/OneDrive-ZHAW/BA/NeurIPs_Paper_Project/seed_f1_results_presentation_with_mean_variance.xlsx";
const input = await FileBlob.load(inputPath);
const workbook = await SpreadsheetFile.importXlsx(input);

for (const spec of [
  { kind: "formula", sheetId: "Presentation", range: "A1:P25", maxChars: 12000, options: { maxResults: 300 } },
  { kind: "formula", sheetId: "All Results", range: "A1:H106", maxChars: 12000, options: { maxResults: 300 } },
  { kind: "computedStyle", sheetId: "Presentation", range: "A1:P8", maxChars: 12000 },
  { kind: "computedStyle", sheetId: "All Results", range: "A1:H10", maxChars: 12000 },
]) {
  const result = await workbook.inspect(spec);
  console.log(`===== ${spec.kind} ${spec.sheetId} =====`);
  console.log(result.ndjson);
}

for (const sheet of workbook.worksheets.items) {
  console.log(`TABLES ${sheet.name}`);
  for (const table of sheet.tables.items) console.log(`${table.name} ${table.range?.address ?? ""}`);
}
