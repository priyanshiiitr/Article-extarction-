"""Phase 10 entry point: score the pipeline against the gold set.

    python scripts/evaluate.py
"""

from __future__ import annotations

from src.config import load_config
from src.entity_resolution.profiles import build_all_profiles
from src.entity_resolution.resolver import resolve_entities
from src.evaluation.evaluate import evaluate, load_gold
from src.logging_utils import configure_logging
from src.preprocessing.processor import read_documents


def main() -> None:
    cfg = load_config()
    configure_logging(cfg.logging.level, cfg.logging.format)

    documents = read_documents(
        cfg.path(cfg.paths.processed_dir) / "documents_relations.jsonl"
    )
    gold = load_gold(cfg.path(cfg.paths.sample_dir) / "gold" / "annotations.json")

    profiles = build_all_profiles(documents)
    entities, _, _ = resolve_entities(profiles, cfg=cfg)

    report = evaluate(documents, gold, entities)

    print("\n" + "=" * 84)
    print(f"EVALUATION REPORT   ({report.articles_evaluated} manually labelled articles)")
    print("=" * 84)

    print("\n  NAMED ENTITY RECOGNITION")
    print("  " + report.ner_strict.row("strict (span + label)"))
    print("  " + report.ner_relaxed.row("relaxed (overlap+label)"))
    print("  " + report.ner_detection_only.row("detection only (no label)"))
    gap = report.ner_relaxed.f1 - report.ner_strict.f1
    print(f"\n    strict->relaxed gap = {gap:+.3f}  "
          f"({'boundary errors dominate' if gap > 0.05 else 'boundaries are fine'})")
    typing_gap = report.ner_detection_only.f1 - report.ner_strict.f1
    print(f"    detection->strict gap = {typing_gap:+.3f}  "
          f"({'typing errors dominate' if typing_gap > 0.05 else 'typing is fine'})")

    print("\n    per label:")
    for label, prf in sorted(report.ner_by_label.items(), key=lambda kv: -kv[1].f1):
        print("      " + prf.row(label))

    print("\n  COREFERENCE")
    print("  " + report.coref_muc.row("MUC"))
    p, r, f = report.coref_b3
    print(f"  {'B-cubed':<26} P={p:.3f}  R={r:.3f}  F1={f:.3f}")
    print("    (MUC rewards over-merging and ignores singletons; B-cubed punishes")
    print("     both over-merging and over-splitting. Read them together.)")

    print("\n  RELATION EXTRACTION")
    print("  " + report.relations.row("all predicates"))
    for predicate, prf in sorted(report.relations_by_predicate.items(), key=lambda kv: -kv[1].f1):
        print("      " + prf.row(predicate))

    if report.entity_resolution:
        er = report.entity_resolution
        print("\n  ENTITY RESOLUTION")
        print("  " + er.pairwise.row("pairwise"))
        print(f"\n    FALSE MERGES : {er.false_merges}   "
              f"<- corrupting, hard to undo")
        for a, b in er.false_merge_examples:
            print(f"        {a}  ==  {b}")
        print(f"    FALSE SPLITS : {er.false_splits}   "
              f"<- costs recall, recoverable")
        for a, b in er.false_split_examples:
            print(f"        {a}  !=  {b}")

    if report.notes:
        print("\n  NOTES")
        for note in report.notes[:12]:
            print(f"    - {note}")

    print("\n" + "=" * 84)
    print("  CAVEAT: 3 articles, single annotator. These are development signals,")
    print("  not a benchmark. Real evaluation needs 2+ annotators and a kappa score.")
    print("=" * 84)


if __name__ == "__main__":
    main()
