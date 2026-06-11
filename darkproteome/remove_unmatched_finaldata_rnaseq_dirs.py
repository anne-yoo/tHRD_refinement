#!/usr/bin/env python3

import argparse
import csv
import shutil
from pathlib import Path


def normalize_text(value):
    if value is None:
        return ""
    return str(value).strip()


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)


def load_linked_riboseq_gsms(manifest_path):
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    linked = set()
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if normalize_text(row.get("status")) != "LINKED":
                continue
            gsm = normalize_text(row.get("riboseq_gsm"))
            if gsm:
                linked.add(gsm)
    return linked


def write_plan(plan_txt_path, plan_tsv_path, rows):
    ensure_dir(plan_txt_path.parent)

    with plan_txt_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(f"{row['dir_path']}\n")

    with plan_tsv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_dir_name", "dir_path", "reason"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)


def make_plan(args):
    organ_dir = args.organ_dir.expanduser().resolve()
    finaldata_dir = organ_dir / "finaldata"
    rnaseq_dir = finaldata_dir / "RNAseq"
    manifests_dir = finaldata_dir / "manifests"
    manifest_path = manifests_dir / "matched_rnaseq_link_manifest.tsv"
    plan_txt_path = manifests_dir / "rnaseq_dirs_to_remove.txt"
    plan_tsv_path = manifests_dir / "rnaseq_dirs_to_remove.tsv"

    if not rnaseq_dir.is_dir():
        raise FileNotFoundError(f"Missing finaldata RNAseq dir: {rnaseq_dir}")

    linked = load_linked_riboseq_gsms(manifest_path)
    rows = []
    for child in sorted(rnaseq_dir.iterdir(), key=lambda p: p.name):
        if not child.is_dir():
            continue
        if child.name in linked:
            continue
        rows.append(
            {
                "sample_dir_name": child.name,
                "dir_path": str(child),
                "reason": "not_present_in_LINKED_riboseq_gsm_set",
            }
        )

    write_plan(plan_txt_path, plan_tsv_path, rows)

    print(f"Organ dir: {organ_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Linked riboseq_gsm count: {len(linked)}")
    print(f"RNAseq sample dirs to remove: {len(rows)}")
    print(f"Plan TXT: {plan_txt_path}")
    print(f"Plan TSV: {plan_tsv_path}")


def apply_plan(args):
    organ_dir = args.organ_dir.expanduser().resolve()
    finaldata_dir = organ_dir / "finaldata"
    rnaseq_dir = finaldata_dir / "RNAseq"
    manifests_dir = finaldata_dir / "manifests"
    plan_txt_path = (
        args.plan_file.expanduser().resolve()
        if args.plan_file
        else manifests_dir / "rnaseq_dirs_to_remove.txt"
    )

    if not plan_txt_path.is_file():
        raise FileNotFoundError(f"Missing plan file: {plan_txt_path}")

    deleted = 0
    skipped = 0
    for raw_line in plan_txt_path.read_text(encoding="utf-8").splitlines():
        dir_path_text = raw_line.strip()
        if not dir_path_text:
            continue

        target = Path(dir_path_text).expanduser().resolve()
        try:
            target.relative_to(rnaseq_dir.resolve())
        except ValueError as exc:
            raise RuntimeError(f"Refusing to delete path outside {rnaseq_dir}: {target}") from exc

        if not target.exists():
            print(f"SKIP_MISSING\t{target}")
            skipped += 1
            continue
        if not target.is_dir():
            print(f"SKIP_NOT_DIR\t{target}")
            skipped += 1
            continue

        print(f"DELETE\t{target}")
        shutil.rmtree(target)
        deleted += 1

    print(f"Plan file: {plan_txt_path}")
    print(f"Deleted dirs: {deleted}")
    print(f"Skipped entries: {skipped}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Two-step safe removal of finaldata/RNAseq sample folders that are not LINKED "
            "in matched_rnaseq_link_manifest.tsv."
        )
    )
    parser.add_argument(
        "organ_dir",
        type=Path,
        help="Organ directory such as /home/.../data/RPFdb/kidney or pancreas",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Delete directories listed in the plan file. Without this flag, only create the plan.",
    )
    parser.add_argument(
        "--plan-file",
        type=Path,
        default=None,
        help=(
            "Plan file to apply. Default: <organ_dir>/finaldata/manifests/rnaseq_dirs_to_remove.txt"
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.apply:
        apply_plan(args)
    else:
        make_plan(args)


if __name__ == "__main__":
    main()
