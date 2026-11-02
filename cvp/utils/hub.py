import os

from huggingface_hub import hf_hub_download


def resolve_file(path: str, repo_id: str) -> str:
    """
    Return `path` if it exists, otherwise download the file of the same name from the Hugging Face repo `repo_id`
    into the folder of `path`.
    """
    if os.path.exists(path):
        return path
    folder = os.path.dirname(path) or "."
    filename = os.path.basename(path)
    print(f"'{path}' not found, downloading '{filename}' from https://huggingface.co/{repo_id}")
    return hf_hub_download(repo_id=repo_id, filename=filename, local_dir=folder)
