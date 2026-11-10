from pathlib import Path
from pypdf import PdfReader, PdfWriter

root = Path("/Users/leonakryeziu/ZHAW_BSc_Modul/SEM6/BA/spectroscopy-qml-xkye-verdiant/outputs/paper-learning-curves-all-seeds-final-20261110")
modalities = ("cnmr", "hnmr", "ir", "msms_neg", "msms_pos")


def merge(inputs, output):
    writer = PdfWriter()
    for item in inputs:
        reader = PdfReader(item)
        for page in reader.pages:
            writer.add_page(page)
    with output.open("wb") as file:
        writer.write(file)


main = root / "01_main_native_axes"
merge([main / f"main_{modality}_native_axes.pdf" for modality in modalities], main / "main_learning_curves_native_axes.pdf")

appendix = root / "04_appendix_individual_seeds"
merge(
    [appendix / f"seed_{seed}" / f"{modality}_seed_{seed}_individual.pdf" for seed in range(41, 46) for modality in modalities],
    appendix / "appendix_individual_learning_curves.pdf",
)

supplement = root / "05_supplement_normalized_qualitative"
merge(
    [supplement / f"seed_{seed}" / f"{modality}_seed_{seed}_normalized_qualitative.pdf" for seed in range(41, 46) for modality in modalities],
    supplement / "supplement_normalized_progress_qualitative.pdf",
)
