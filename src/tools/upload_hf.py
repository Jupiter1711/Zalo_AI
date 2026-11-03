"""Đẩy dữ liệu đã xử lý (+ code) lên một Hugging Face dataset repo PRIVATE để notebook trên molab/Kaggle tải về.

Không upload dữ liệu cuộc thi qua file browser của molab: notebook molab mặc định là public (không ai tìm thấy nhưng ai có
link đều xem/fork được, kèm file đã upload). Notebook chỉ giữ HF_TOKEN trong .env (không bị fork) và tải dữ liệu lúc chạy.

Dữ liệu gốc (--with-raw) được đóng gói thành vài file tar trong packs/ thay vì hàng nghìn file lẻ: HF giới hạn
1000 request / 5 phút, tải từng file state.json/wav sẽ bị lỗi 429.
    packs/<split>_core.tar.gz   jsonl, tools_catalog, zaic_tools, <split>_environment
    packs/<split>_audio.tar     audio/ (wav không nén thêm được)

Cài và đăng nhập (một lần, trong venv của repo):
    pip install huggingface_hub
    hf auth login                    # huggingface_hub >= 1.0 dùng lệnh `hf`; bản cũ < 1.0 là `huggingface-cli login`
                                     # hoặc đặt biến môi trường HF_TOKEN (token quyền write)
Chạy từ gốc repo:
    python -m src.tools.upload_hf --repo <hf_user>/zalo-lifemate-data              # sft + asr manifest + splits + src (~30MB)
    python -m src.tools.upload_hf --repo <hf_user>/zalo-lifemate-data --with-raw   # thêm packs/ dữ liệu gốc (~1GB, cần cho inference)
    python -m src.tools.upload_hf --pack-only --pack-dir <thư mục>                 # chỉ tạo file tar ở local, không upload
"""
from __future__ import annotations

import argparse
import os
import tarfile
import tempfile
from pathlib import Path

from src.data.common import DEFAULT_DATA_ROOT

IGNORE = ["**/__pycache__/**", "*.pyc", "**/.DS_Store"]
SPLITS = ("train", "public_test")


def _skip(path: Path) -> bool:
    return "__pycache__" in path.parts or path.suffix == ".pyc" or path.name == ".DS_Store"


def make_packs(data_root: Path, out_dir: Path) -> list[Path]:
    """Tạo packs/<split>_core.tar.gz và packs/<split>_audio.tar; đường dẫn trong tar tương đối với data_root."""
    out_dir.mkdir(parents=True, exist_ok=True)
    packs = []
    for split in SPLITS:
        base = data_root / split
        if not base.exists():
            print(f"bỏ qua {split}: không có {base}")
            continue
        files = sorted(p for p in base.rglob("*") if p.is_file() and not _skip(p))
        core = [p for p in files if p.relative_to(base).parts[0] != "audio"]
        audio = [p for p in files if p.relative_to(base).parts[0] == "audio"]
        for name, group, mode in ((f"{split}_core.tar.gz", core, "w:gz"), (f"{split}_audio.tar", audio, "w")):
            path = out_dir / name
            with tarfile.open(path, mode) as tar:
                for p in group:
                    tar.add(p, arcname=p.relative_to(data_root).as_posix())
            print(f"{name}: {len(group)} file, {path.stat().st_size / 2**20:.0f} MB")
            packs.append(path)
    return packs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", help="<hf_user>/<tên repo dataset>")
    ap.add_argument("--with-raw", action="store_true", help="đóng gói + upload dữ liệu gốc (audio, state, zaic_tools) vào packs/")
    ap.add_argument("--pack-only", action="store_true", help="chỉ tạo file tar vào --pack-dir, không upload gì")
    ap.add_argument("--pack-dir", type=Path, help="thư mục chứa file tar (mặc định: thư mục tạm)")
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    args = ap.parse_args()

    if args.pack_only:
        if not args.pack_dir:
            ap.error("--pack-only cần --pack-dir")
        make_packs(args.data_root, args.pack_dir)
        return
    if not args.repo:
        ap.error("cần --repo")

    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(args.repo, repo_type="dataset", private=True, exist_ok=True)
    if not api.repo_info(args.repo, repo_type="dataset").private:
        raise SystemExit(f"DỪNG: repo {args.repo} đang PUBLIC. Đổi sang private trên huggingface.co rồi chạy lại.")

    uploads = [
        ("data/processed/sft", "sft"),
        ("data/processed/sft_v2", "sft_v2"),  # build_sft --mix (text sạch + text ASR k-fold)
        ("data/processed/asr", "asr"),
        ("splits", "splits"),
        ("src", "src"),
    ]
    for local, remote in uploads:
        if not Path(local).exists():
            print(f"bỏ qua {local}: không tồn tại")
            continue
        print(f"upload {local} -> {args.repo}/{remote}")
        api.upload_folder(repo_id=args.repo, repo_type="dataset", folder_path=local, path_in_repo=remote,
                          ignore_patterns=IGNORE, commit_message=f"upload {remote}")
    if args.with_raw:
        with tempfile.TemporaryDirectory() as tmp:
            pack_dir = args.pack_dir or Path(tmp)
            make_packs(args.data_root, pack_dir)
            print(f"upload {pack_dir} -> {args.repo}/packs")
            api.upload_folder(repo_id=args.repo, repo_type="dataset", folder_path=str(pack_dir), path_in_repo="packs",
                              allow_patterns=["*.tar", "*.tar.gz"], commit_message="upload packs")
    print(f"xong: https://huggingface.co/datasets/{args.repo} (private)")


if __name__ == "__main__":
    main()
