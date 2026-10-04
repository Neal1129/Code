from pathlib import Path
import csv
import os
import sys
import zipfile
import xml.etree.ElementTree as ET

PYTHON = "/opt/anaconda3/envs/esm2/bin/python"
if sys.executable != PYTHON:
    os.execv(PYTHON, [PYTHON, __file__, *sys.argv[1:]])

import esm
import matplotlib.pyplot as plt
import torch


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data" / "esm2" / "results-table.xlsx"
OUTPUT = ROOT / "results" / "esm2"
OUTPUT.mkdir(exist_ok=True)
MODEL = "esm2_t33_650M_UR50D"
WT = "MMQPSIKPADEHSAGDIIARIGSLTRMLRDSLRELGLDQAIAEAAEAIPDARDRLYYVVQMTAQAAERALNSVEASQPHQDQMEKSAKALTQRWDDWFADPIDLADARELVTDTRQFLADVPAHTSFTNAQLLKIMMAQDFQDLTGQVIKRMMDVIQEIERQLLMVLLENIPEQESRPKRENQSLLNGPQVDTSKGGSSSLATTLERIEKNFVITDPRLPDNPIIFASDSFLQLTEYSREEILGRNCRFLQGPETDRATVRKIRDAIDNQTEVTVQLINYTKSGKKFWNVFHLQPMRDYKGDVQYFIGVQLDGTERGSAGVVASQDQVDDLLDSLGF"
LINKER_START = WT.index("GGSSS", 190)
BATCH_SIZE = 4


def progress(text):
    print(f"\033[92m{text}\033[0m", flush=True)


def read_results_table():
    namespace = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    with zipfile.ZipFile(INPUT) as workbook:
        shared = ET.fromstring(workbook.read("xl/sharedStrings.xml"))
        strings = ["".join(item.itertext()) for item in shared]
        sheet = ET.fromstring(workbook.read("xl/worksheets/sheet1.xml"))
    rows = []
    for row in sheet.findall(f".//{namespace}row"):
        values = []
        for cell in row.findall(f"{namespace}c"):
            value = cell.find(f"{namespace}v")
            text = "" if value is None else value.text
            values.append(strings[int(text)] if cell.get("t") == "s" else text)
        rows.append(values)
    return [dict(zip(rows[0], row)) for row in rows[1:]]


def score_site(site, model, alphabet, data):
    batch_converter = alphabet.get_batch_converter()
    position = LINKER_START + site

    sequences = [WT] + list(dict.fromkeys(row["Sequence"] for row in data))
    masked = []
    for sequence_id, sequence in enumerate(sequences):
        masked_sequence = list(sequence)
        masked_sequence[position] = "<mask>"
        masked.append((sequence_id, position, "".join(masked_sequence)))

    log_probabilities = {}
    for first in range(0, len(masked), BATCH_SIZE):
        batch = masked[first:first + BATCH_SIZE]
        _, _, tokens = batch_converter([(f"{sid}_{position}", sequence) for sid, position, sequence in batch])
        with torch.inference_mode():
            logits = model(tokens, repr_layers=[], return_contacts=False)["logits"]
        log_probs = torch.log_softmax(logits, dim=-1)
        for i, (sequence_id, position, _) in enumerate(batch):
            log_probabilities[sequence_id, position] = log_probs[i, position + 1].cpu()
        done = min(first + BATCH_SIZE, len(masked))
        bar = "█" * round(done / len(masked) * 20) + "·" * (20 - round(done / len(masked) * 20))
        progress(f"  位点 {site + 1}/5  [{bar}]  {done}/{len(masked)} 条序列  {done / len(masked):.0%}")

    wt_token = alphabet.get_idx(WT[position])
    sequence_scores = {}
    for sequence_id, sequence in enumerate(sequences[1:], 1):
        mutant_token = alphabet.get_idx(sequence[position])
        sequence_scores[sequence] = log_probabilities[sequence_id, position][mutant_token].item() - log_probabilities[0, position][wt_token].item()

    with open(OUTPUT / f"esm2_site_{site}.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Sequence", "score"])
        writer.writerows(sequence_scores.items())


def assemble():
    data = read_results_table()
    scores = {}
    for site in range(5):
        with open(OUTPUT / f"esm2_site_{site}.csv") as file:
            for row in list(csv.DictReader(file)):
                scores[row["Sequence"], site] = float(row["score"])

    rows = []
    for row in data:
        sequence = row["Sequence"]
        score = sum(scores[sequence, site] for site in range(5))
        rows.append({
            "Original_rank": int(row["Rank"]),
            "AI_score": float(row["Score"]),
            "Linker": sequence[LINKER_START:LINKER_START + 5],
            "ESM2_masked_marginal_score": score,
            "ESM2_mean_per_site": score / 5,
            "Sequence": sequence,
        })

    results = sorted(rows, key=lambda row: row["ESM2_masked_marginal_score"], reverse=True)
    for rank, row in enumerate(results, 1):
        row["ESM2_rank"] = rank
    columns = ["ESM2_rank", "Original_rank", "AI_score", "Linker", "ESM2_masked_marginal_score", "ESM2_mean_per_site", "Sequence"]
    with open(OUTPUT / "esm2_masked_marginal_results.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(results)

    plt.rcParams.update({"font.family": "Arial", "font.size": 9})
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    ax.scatter([r["AI_score"] for r in results], [r["ESM2_masked_marginal_score"] for r in results], s=28, color="#0072B2")
    offsets = [(8, 10), (10, 8), (8, 10), (8, -18), (10, 12)]
    annotated = set()
    for r in results:
        if r["Linker"] not in annotated:
            ax.annotate(r["Linker"], (r["AI_score"], r["ESM2_masked_marginal_score"]), xytext=offsets[len(annotated)], textcoords="offset points", fontsize=7, arrowprops={"arrowstyle": "-", "color": "0.35", "lw": 0.6})
            annotated.add(r["Linker"])
        if len(annotated) == 5:
            break
    ax.axhline(0, color="0.55", linewidth=0.8)
    ax.set_ylim(top=4.65)
    ax.set_xlabel("Input AI score")
    ax.set_ylabel("ESM-2 masked-marginal score\n(relative to wild-type GGSSS linker)")
    fig.tight_layout()
    fig.savefig(OUTPUT / "esm2_masked_marginal_vs_ai_score.jpg", dpi=600)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "assemble":
        assemble()
    else:
        progress("ESM-2 模型加载中…")
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        model.eval()
        torch.set_num_threads(10)
        data = read_results_table()
        sites = range(5) if len(sys.argv) == 1 else [int(sys.argv[1])]
        for site in sites:
            progress(f"\n开始计算连接肽位点 {site + 1}/5…")
            score_site(site, model, alphabet, data)
        if len(sys.argv) == 1:
            progress("\n正在汇总结果并绘制 600 ppi 图片…")
            assemble()
            progress(f"完成：{OUTPUT / 'esm2_masked_marginal_results.csv'}")
            progress(f"完成：{OUTPUT / 'esm2_masked_marginal_vs_ai_score.jpg'}")
