import marimo

__generated_with = "0.16.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
    # LifeMate: fine-tune ASR (PhoWhisper-large)

    Fine-tune toàn bộ **vinai/PhoWhisper-large** (1,55B tham số) trên audio train có transcript (manifest từ
    `src/data/build_asr_manifest.py`). Nhãn là `text_asr`: chữ thường, không dấu câu, số đọc thành chữ; đây là đúng dạng mà
    `llm_input` (ITN) nhận ở inference.

    - Mặc định train trên `train` + `val_seen`, đo WER trên `val_unseen` (app chưa thấy, nên tên riêng cũng lạ hơn).
    - Bản nộp cuối: `train_splits=train,val_seen,val_unseen` (WER đo lại trên val_unseen lúc đó sẽ lạc quan).
    - Model fp16 được đẩy lên `out_repo` (private). Sau đó chạy `infer.py asr_model=<out_repo>`.

    Luật: tổng tham số các model < 6B. Qwen3-4B (~4,0B) + PhoWhisper-large (~1,55B) ≈ 5,6B.

    Chạy: `python -u train_asr.py` (tham số `key=value` để ghi đè cấu hình, vd. `epochs=5 lr=5e-6`).
    """
    )
    return


@app.cell
def _():
    from sys import argv as _argv

    CFG = dict(
        base_model="vinai/PhoWhisper-large",
        data_repo="Jupiter1711/zalo-lifemate-data",
        out_repo="Jupiter1711/lifemate-phowhisper-large-ft",  # HF model repo private; "" = không push
        train_splits="train,val_seen",  # bản cuối: train,val_seen,val_unseen
        eval_split="val_unseen",
        eval_before=True,  # đo WER model gốc trên cùng tập để so sánh
        epochs=4,
        lr=1e-5,
        batch_size=16,
        grad_accum=1,
        warmup_steps=50,
        weight_decay=0.01,
        spec_augment=True,  # che ngẫu nhiên một phần phổ khi train, chống overfit với 5 giờ audio
        freeze_encoder=False,
        eval_bs=32,
        seed=2026,
        work_dir="/tmp/lifemate",
        # Chế độ k-fold (fold >= 0): gộp train+val_seen+val_unseen, chia theo sample thành n_folds phần; train trên các
        # phần khác rồi chép phần `fold`. Transcript này có lỗi giống ASR thật (model chưa nghe audio đó) để làm dữ liệu
        # SFT cho LLM chịu được lỗi ASR. Không đẩy model, chỉ đẩy transcript lên data_repo outputs/asr_folds/.
        fold=-1,
        n_folds=2,
    )
    for _arg in _argv[1:]:
        if "=" not in _arg or _arg.startswith("-"):
            continue
        _k, _v = _arg.split("=", 1)
        if _k not in CFG:
            raise SystemExit(f"tham số lạ: {_k}. Có: {', '.join(CFG)}")
        _d = CFG[_k]
        CFG[_k] = ((_v.lower() in ("1", "true", "yes")) if isinstance(_d, bool) else type(_d)(_v)
                   if isinstance(_d, (int, float)) else _v)
    if CFG["fold"] >= 0:
        assert CFG["fold"] < CFG["n_folds"], "fold phải < n_folds"
        CFG["eval_before"] = False
    print("CFG:", CFG)
    return (CFG,)


@app.cell
def _():
    import os as _os
    import subprocess as _sp
    import sys as _sys

    # molab cài sẵn TensorFlow + Keras 3: tắt TF trước khi import transformers.
    _os.environ["USE_TF"] = "0"
    _os.environ["USE_TORCH"] = "1"
    _os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    _pkgs = ["transformers==4.57.1", "accelerate==1.10.1", "huggingface_hub>=0.34,<1.0"]
    _sp.run([_sys.executable, "-m", "pip", "install", "-q", *_pkgs], check=True)

    import torch
    import transformers

    print("torch", torch.__version__, "| transformers", transformers.__version__)
    assert torch.cuda.is_available(), "Chưa có GPU: bật GPU ở nút specs của notebook"
    _p = torch.cuda.get_device_properties(0)
    BF16 = torch.cuda.is_bf16_supported()
    print(f"GPU: {_p.name} | VRAM {_p.total_memory / 2**30:.0f} GB | bf16: {BF16}")
    return BF16, torch


@app.cell
def _():
    import json
    import os
    import time
    from pathlib import Path

    _env = Path(".env")
    if _env.exists():
        for _line in _env.read_text(encoding="utf-8").splitlines():
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip().strip("'\""))
    HF_TOKEN = os.environ.get("HF_TOKEN")
    assert HF_TOKEN, "Thiếu HF_TOKEN: tạo file .env với dòng HF_TOKEN=hf_..."
    return HF_TOKEN, Path, json, time


@app.cell
def _(mo):
    run_btn = mo.ui.run_button(label="Train ASR")
    run_btn
    return (run_btn,)


@app.cell
def _(CFG, HF_TOKEN, Path, json, mo, run_btn, time, torch):
    mo.stop(not (run_btn.value or mo.app_meta().mode == "script"), mo.md("Bấm **Train ASR** để bắt đầu."))
    import sys
    import tarfile
    import wave

    import numpy as np
    from huggingface_hub import hf_hub_download, snapshot_download

    assert torch is not None
    _t = time.time()
    DATA_DIR = Path(snapshot_download(CFG["data_repo"], repo_type="dataset", token=HF_TOKEN,
                                      local_dir=f"{CFG['work_dir']}/data", allow_patterns=["src/*", "asr/*"]))
    if str(DATA_DIR) not in sys.path:
        sys.path.insert(0, str(DATA_DIR))
    # audio train: một file tar (tải từng wav sẽ vượt giới hạn request của HF)
    RAW = Path(CFG["work_dir"]) / "raw"
    _name = "train_audio.tar"
    _tar = Path(hf_hub_download(CFG["data_repo"], f"packs/{_name}", repo_type="dataset", token=HF_TOKEN,
                                local_dir=f"{CFG['work_dir']}/dl"))
    _done = RAW / f".{_name}.done"
    _stamp = f"{_tar.stat().st_size}:{_tar.stat().st_mtime_ns}"
    if not (_done.exists() and _done.read_text() == _stamp):
        with tarfile.open(_tar) as _tf:
            _tf.extractall(RAW, filter="data")
        _done.write_text(_stamp)

    def read_wav(rel):
        with wave.open(str(RAW / rel)) as w:
            assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (16000, 1, 2), rel
            return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768

    def load_manifest(split):
        with open(DATA_DIR / "asr" / f"{split}.jsonl", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        for r in rows:
            r["array"] = read_wav(r["audio"])
        return rows

    if CFG["fold"] < 0:
        train_rows = [r for _s in CFG["train_splits"].split(",") for r in load_manifest(_s.strip())]
        eval_rows = load_manifest(CFG["eval_split"])
    else:  # k-fold theo sample: mọi lượt của một sample nằm cùng một phần
        import hashlib

        _pool = [r for _s in ("train", "val_seen", "val_unseen") for r in load_manifest(_s)]

        def _fold_of(sid):
            return int(hashlib.md5(sid.encode()).hexdigest(), 16) % CFG["n_folds"]

        train_rows = [r for r in _pool if _fold_of(r["sample_id"]) != CFG["fold"]]
        eval_rows = [r for r in _pool if _fold_of(r["sample_id"]) == CFG["fold"]]
        print(f"k-fold {CFG['fold']}/{CFG['n_folds']}: train trên các phần khác, chép phần này")
    _hours = sum(len(r["array"]) for r in train_rows) / 16000 / 3600
    print(f"train: {len(train_rows)} audio ({_hours:.2f} giờ) từ {CFG['train_splits']} | "
          f"eval: {len(eval_rows)} audio ({CFG['eval_split']}) | tải + đọc {time.time() - _t:.0f}s")
    return eval_rows, train_rows


@app.cell
def _(BF16, CFG, torch, train_rows):
    from transformers import WhisperForConditionalGeneration, WhisperProcessor, set_seed

    assert train_rows  # chờ ô dữ liệu (nút Train ASR) rồi mới nạp model
    set_seed(CFG["seed"])
    processor = WhisperProcessor.from_pretrained(CFG["base_model"])
    processor.tokenizer.set_prefix_tokens(language="vietnamese", task="transcribe", predict_timestamps=False)
    model = WhisperForConditionalGeneration.from_pretrained(CFG["base_model"], dtype=torch.float32).to("cuda")
    model.generation_config.forced_decoder_ids = None  # dùng language/task khi generate
    model.config.forced_decoder_ids = None
    if CFG["spec_augment"]:
        model.config.apply_spec_augment = True
        model.config.mask_time_prob = 0.05
        model.config.mask_time_length = 10
        model.config.mask_feature_prob = 0.05
        model.config.mask_feature_length = 10
    if CFG["freeze_encoder"]:
        model.model.encoder.requires_grad_(False)
    print(f"{CFG['base_model']}: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B tham số, "
          f"train {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e9:.2f}B | autocast "
          f"{'bf16' if BF16 else 'fp16'}")
    return model, processor


@app.cell
def _(BF16, CFG, eval_rows, model, processor, torch):
    from src.text.vi_numbers import asr_target  # src có trong sys.path sau ô dữ liệu (eval_rows)

    assert eval_rows

    def edit_distance(r, h):
        d = list(range(len(h) + 1))
        for i in range(1, len(r) + 1):
            prev, d[0] = d[0], i
            for j in range(1, len(h) + 1):
                prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
        return d[len(h)]

    def transcribe(arrays, bs=None):
        """Greedy, ép tiếng Việt; trả về text thô theo thứ tự đầu vào."""
        bs = bs or CFG["eval_bs"]
        model.eval()
        order = sorted(range(len(arrays)), key=lambda i: len(arrays[i]))
        out = [""] * len(arrays)
        dtype = torch.bfloat16 if BF16 else torch.float16
        for s in range(0, len(order), bs):
            idx = order[s: s + bs]
            feats = processor.feature_extractor([arrays[i] for i in idx], sampling_rate=16000,
                                                return_tensors="pt").input_features.to("cuda")
            with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
                ids = model.generate(feats, language="vi", task="transcribe", max_new_tokens=200)
            for i, t in zip(idx, processor.batch_decode(ids, skip_special_tokens=True)):
                out[i] = t.strip()
        return out

    def eval_wer(rows, tag, show=8):
        """WER sau chuẩn hóa asr_target (chữ thường, bỏ dấu câu, số -> chữ), cả tập và riêng lượt cuối."""
        hyps = transcribe([r["array"] for r in rows])
        stats = {"all": [0, 0], "final": [0, 0]}
        bad = []
        for r, h in zip(rows, hyps):
            ref, hyp = r["text_asr"].split(), asr_target(h).split()
            e = edit_distance(ref, hyp)
            for k in ("all", "final") if r["is_final"] else ("all",):
                stats[k][0] += e
                stats[k][1] += len(ref)
            if e:
                bad.append((e / max(len(ref), 1), r["id"], r["text_asr"], asr_target(h)))
        res = {k: round(a / max(b, 1), 4) for k, (a, b) in stats.items()}
        exact = sum(asr_target(h) == r["text_asr"] for r, h in zip(rows, hyps)) / len(rows)
        print(f"[{tag}] WER tất cả lượt {res['all']:.2%} | WER lượt cuối {res['final']:.2%} | "
              f"đúng nguyên câu {exact:.1%} | {len(rows)} audio")
        for _, sid, ref, hyp in sorted(bad, reverse=True)[:show]:
            print(f"  {sid}\n    ref: {ref}\n    hyp: {hyp}")
        return res, hyps

    return eval_wer, transcribe


@app.cell
def _(BF16, CFG, HF_TOKEN, Path, eval_rows, eval_wer, json, model, processor, time, torch, train_rows):
    import random

    from transformers import Trainer, TrainerCallback, TrainingArguments, PrinterCallback

    if CFG["eval_before"]:
        wer_before = eval_wer(eval_rows, "model gốc")[0]

    _labels = {r["id"]: processor.tokenizer(r["text_asr"]).input_ids for r in train_rows}
    print("nhãn mẫu:", processor.tokenizer.decode(_labels[train_rows[0]["id"]]))
    train_ds = [{"array": r["array"], "labels": _labels[r["id"]]} for r in train_rows]
    random.Random(CFG["seed"]).shuffle(train_ds)
    _start_id = model.config.decoder_start_token_id
    _fe, _tok = processor.feature_extractor, processor.tokenizer

    def _collate(batch):
        feats = _fe([b["array"] for b in batch], sampling_rate=16000, return_tensors="pt").input_features
        lab = _tok.pad({"input_ids": [b["labels"] for b in batch]}, return_tensors="pt")
        labels = lab["input_ids"].masked_fill(lab["attention_mask"].ne(1), -100)
        if (labels[:, 0] == _start_id).all():  # model tự thêm decoder_start_token
            labels = labels[:, 1:]
        return {"input_features": feats, "labels": labels}

    class _LogCb(TrainerCallback):
        def on_train_begin(self, args, state, control, **kw):
            self.t0 = time.time()

        def on_log(self, args, state, control, logs=None, **kw):
            if not logs or "loss" not in logs:
                return
            spent = time.time() - self.t0
            eta = spent / max(state.global_step, 1) * (state.max_steps - state.global_step)
            print(f"step {state.global_step}/{state.max_steps} | epoch {logs.get('epoch', 0):.2f} | "
                  f"loss {logs['loss']:.4f} | lr {logs.get('learning_rate', 0):.2e} | "
                  f"đã chạy {spent / 60:.1f}' còn ~{eta / 60:.1f}'", flush=True)

    _ckpt = f"{CFG['work_dir']}/asr_ckpt"
    _args = TrainingArguments(
        output_dir=_ckpt,
        per_device_train_batch_size=CFG["batch_size"],
        gradient_accumulation_steps=CFG["grad_accum"],
        num_train_epochs=CFG["epochs"],
        learning_rate=CFG["lr"],
        warmup_steps=CFG["warmup_steps"],
        weight_decay=CFG["weight_decay"],
        lr_scheduler_type="linear",
        bf16=BF16,
        fp16=not BF16,
        logging_steps=10,
        save_strategy="no",  # train ngắn (~vài chục phút): chỉ lưu model cuối
        report_to="none",
        disable_tqdm=True,
        remove_unused_columns=False,
        dataloader_num_workers=3,
        seed=CFG["seed"],
    )
    trainer = Trainer(model=model, args=_args, train_dataset=train_ds, data_collator=_collate)
    trainer.remove_callback(PrinterCallback)
    trainer.add_callback(_LogCb())
    _t = time.time()
    trainer.train()
    print(f"train xong trong {(time.time() - _t) / 60:.1f} phút")

    wer_after, _hyps = eval_wer(eval_rows, "sau fine-tune" if CFG["fold"] < 0 else f"fold {CFG['fold']} (audio chưa nghe)")

    if CFG["fold"] >= 0:
        # {đường dẫn audio như trong <split>.jsonl ("audio/<id>/<file>.wav"): transcript thô} cho build_sft --transcripts
        from huggingface_hub import HfApi as _Api

        _tr = {r["audio"].split("/", 1)[1]: h for r, h in zip(eval_rows, _hyps)}
        _rel = f"outputs/asr_folds/n{CFG['n_folds']}_fold{CFG['fold']}.json"
        _p = Path(CFG["work_dir"]) / _rel
        _p.parent.mkdir(parents=True, exist_ok=True)
        _p.write_text(json.dumps({"cfg": CFG, "wer": wer_after, "transcripts": _tr}, ensure_ascii=False, indent=0),
                      encoding="utf-8")
        _Api(token=HF_TOKEN).upload_file(path_or_fileobj=str(_p), path_in_repo=_rel, repo_id=CFG["data_repo"],
                                         repo_type="dataset", commit_message=f"ASR fold {CFG['fold']}")
        print(f"đã đẩy {len(_tr)} transcript lên {CFG['data_repo']}/{_rel} (k-fold: không lưu model)")

    # lưu fp16 (3 GB thay vì 6 GB) + processor, đẩy lên HF private (chế độ k-fold không lưu model)
    _out = Path(CFG["work_dir"]) / "asr_ft"
    if CFG["fold"] < 0:
        model.to(torch.float16).save_pretrained(_out, safe_serialization=True)
        processor.save_pretrained(_out)
        (_out / "train_info.json").write_text(json.dumps({
            "cfg": CFG, "wer_before": wer_before if CFG["eval_before"] else None, "wer_after": wer_after,
            "log_history": trainer.state.log_history}, ensure_ascii=False, indent=1), encoding="utf-8")
    if CFG["out_repo"] and CFG["fold"] < 0:
        from huggingface_hub import HfApi

        _api = HfApi(token=HF_TOKEN)
        _api.create_repo(CFG["out_repo"], private=True, exist_ok=True)
        assert _api.repo_info(CFG["out_repo"]).private, "out_repo phải private"
        _api.upload_folder(repo_id=CFG["out_repo"], folder_path=str(_out),
                           commit_message=f"ASR fine-tune: WER val {wer_after}")
        print(f"đã đẩy lên https://huggingface.co/{CFG['out_repo']} (private)")
    return


if __name__ == "__main__":
    app.run()
