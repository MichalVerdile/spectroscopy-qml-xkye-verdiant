import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const path = "/Users/leonakryeziu/Library/CloudStorage/OneDrive-ZHAW/BA/NeurIPs_Paper_Project/seed_f1_results_presentation.xlsx";
const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(path));
const overview = await workbook.inspect({
  kind: "workbook,sheet,table",
  maxChars: 10000,
  tableMaxRows: 12,
  tableMaxCols: 14,
  tableMaxCellChars: 80,
});
console.log(overview.ndjson);
const styles = await workbook.inspect({
  kind: "computedStyle",
  sheetId: "Presentation",
  range: "A1:L24",
  maxChars: 8000,
});
console.log(styles.ndjson);
const preview = await workbook.render({
  sheetName: "Presentation",
  range: "A1:L25",
  scale: 1.5,
  format: "png",
});
await fs.writeFile(
  "/Users/leonakryeziu/ZHAW_BSc_Modul/SEM6/BA/spectroscopy-qml-xkye-verdiant/.codex-build/rawdata-excel/reference.png",
  new Uint8Array(await preview.arrayBuffer()),
);
