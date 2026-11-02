import argparse
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi


def main():
    parser = argparse.ArgumentParser(description="Upload the checkpoint, the data and the model card to the Hugging Face Hub.")
    parser.add_argument("--repo_id", required=True, help="repository id, e.g. comp-visual-planning/toy-example")
    parser.add_argument("--files", nargs="+", default=["ckpts/diffusion_420000.ckpt", "data/toy.npy"],
                        help="local files to upload, stored under their base names")
    parser.add_argument("--card", default="scripts/model_card.md", help="markdown model card uploaded as README.md")
    parser.add_argument("--message", default="Upload toy example checkpoint and data", help="commit message")
    args = parser.parse_args()

    api = HfApi()
    print(f"logged in as {api.whoami()['name']}")
    url = api.create_repo(args.repo_id, repo_type="model", exist_ok=True)

    operations = [CommitOperationAdd(path_in_repo=Path(f).name, path_or_fileobj=f) for f in args.files]
    operations.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=args.card))
    api.create_commit(repo_id=args.repo_id, repo_type="model", operations=operations, commit_message=args.message)
    print(f"uploaded {[Path(f).name for f in args.files]} and README.md to {url}")


if __name__ == "__main__":
    main()
