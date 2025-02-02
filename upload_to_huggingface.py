from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path="./fine_tuned_deepseek",
    repo_id="tawrid/deepseek-coder-finetuned"
)
print("✅ Model published to Hugging Face!")