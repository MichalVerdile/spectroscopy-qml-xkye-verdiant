import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const input = await FileBlob.load("/Users/leonakryeziu/Library/CloudStorage/OneDrive-ZHAW/BA/NeurIPs_Paper_Project/seed_f1_results_presentation_with_mean_variance.xlsx");
const workbook = await SpreadsheetFile.importXlsx(input);
for (const address of ["A7:H9", "A12:H14", "A17:H19", "A22:H24", "A27:H29"]) {
  const result = await workbook.inspect({ kind: "computedStyle", sheetId: "All Results", range: address, maxChars: 8000 });
  console.log(`===== ${address} =====`);
  console.log(result.ndjson);
}
