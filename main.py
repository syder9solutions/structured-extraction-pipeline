"""CLI entrypoint: extract | batch | eval | serve | config."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.batch import batch_extract


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--provider", choices=["auto", "openai"], default="auto")
    parser.add_argument("--threshold", type=float, default=0.7)


def main() -> None:
    parser = argparse.ArgumentParser(description="Structured Extraction Pipeline")
    parser.add_argument(
        "command",
        nargs="?",
        default="extract",
        choices=["extract", "batch", "eval", "serve", "config"],
    )
    parser.add_argument("--file", type=Path, help="Document path (.txt/.md/.pdf)")
    parser.add_argument("--text", help="Document text inline")
    parser.add_argument(
        "--type", dest="doc_type", default="invoice", choices=["invoice", "receipt"]
    )
    parser.add_argument("--folder", type=Path, help="Folder of documents for batch mode")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8097)
    _add_common(parser)
    args = parser.parse_args()

    if args.command == "config":
        from src.models import Invoice, Receipt

        print(
            json.dumps(
                {
                    "models": {
                        "invoice": Invoice.model_json_schema(),
                        "receipt": Receipt.model_json_schema(),
                    }
                },
                indent=2,
            )
        )
        return

    if args.command == "eval":
        from evals.extraction_eval import run

        report = run()
        print(report.render())
        raise SystemExit(0 if report.passed else 1)

    if args.command == "serve":
        import uvicorn

        uvicorn.run("src.api:app", host=args.host, port=args.port, log_level="info")
        return

    if args.command == "batch":
        if not args.folder or not args.folder.is_dir():
            parser.error("--folder pointing to a directory is required for batch")
        from src.parsing import parse_document

        docs = [
            (parse_document(path), args.doc_type)
            for path in sorted(args.folder.iterdir())
            if path.suffix.lower() in {".txt", ".md", ".pdf"}
        ]
        result = batch_extract(docs)
        print(json.dumps(result.as_dict(), indent=2))
        for item in result.results:
            if item.document:
                print(json.dumps(item.as_dict(), indent=2))
        return

    # extract
    if args.file:
        from src.parsing import parse_document

        text = parse_document(args.file)
    elif args.text:
        text = args.text
    else:
        parser.error("provide --file or --text for extract")

    from src.service import ExtractionService

    service = ExtractionService(review_threshold=args.threshold)
    result = service.extract_text(text, args.doc_type)
    print(json.dumps(result.as_dict(), indent=2))


if __name__ == "__main__":
    main()
