from huggingface_hub import HfApi, create_repo
import os

token = os.environ["HF_TOKEN"]
repo_id = "srijayadav/llama3-8b-qlora-alpaca"
adapter_dir = "outputs/qlora-alpaca/final_adapter"

api = HfApi()

# 1. Create the repo (safe to run if it already exists)
create_repo(repo_id, token=token, exist_ok=True, private=False)
print(f"Repo ready: https://huggingface.co/{repo_id}")

# 2. Upload all files in the adapter directory
api.upload_folder(
    folder_path=adapter_dir,
    repo_id=repo_id,
    token=token,
    commit_message="Add QLoRA adapter weights, tokenizer, and model card",
)
print("Upload complete!")
print(f"View at: https://huggingface.co/{repo_id}")
