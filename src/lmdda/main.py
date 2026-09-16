from __future__ import annotations

from .args import parse_args


def main() -> None:
    args = parse_args()
    if args.mode == "train":
        from .train import run_train_cv

        run_train_cv(args)
    elif args.mode == "export_embeddings":
        from .export_embeddings import run_export_embeddings

        run_export_embeddings(args)
    else:
        from .predict import run_predict

        run_predict(args)


if __name__ == "__main__":
    main()
