import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const source = "/Users/leonakryeziu/Library/CloudStorage/OneDrive-ZHAW/BA/NeurIPs_Paper_Project/raw_data_seed41_f1_results.xlsx";
const outDir = "/Users/leonakryeziu/ZHAW_BSc_Modul/SEM6/BA/spectroscopy-qml-xkye-verdiant/outputs/raw-seed41-training-time-20261109";
const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(source));
const summary = await workbook.inspect({
  kind: "workbook,sheet,table",
  maxChars: 8000,
  tableMaxRows: 30,
  tableMaxCols: 10,
});
console.log(summary.ndjson);
for (const sheetInfo of workbook.worksheets.items) {
  const sheet = workbook.worksheets.getItem(sheetInfo.name);
  const used = sheet.getUsedRange();
  console.log(`SHEET=${sheetInfo.name} USED=${used?.address ?? "none"}`);
  if (used) {
    const style = await workbook.inspect({
      kind: "computedStyle",
      sheetId: sheetInfo.name,
      range: used.address,
      maxChars: 6000,
    });
    console.log(style.ndjson);
    const rendered = await workbook.render({ sheetName: sheetInfo.name, range: used.address, scale: 2, format: "png" });
    await fs.writeFile(`${outDir}/source_${sheetInfo.name.replaceAll(" ", "_")}.png`, new Uint8Array(await rendered.arrayBuffer()));
  }
}
