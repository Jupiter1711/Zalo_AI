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
    # LifeMate: inference (ASR → ITN → agent) + chấm điểm / tạo file nộp

    Pipeline: audio → **PhoWhisper** (zero-shot) → `llm_input` (chuẩn hóa + ITN số) → **Qwen3-4B + LoRA** gọi tool trên
    `ToolEnv` của BTC (vòng agent trong `src/agent/loop.py`) → `submission.jsonl`.

    **Chạy** (trong ô terminal/subprocess như notebook train), cấu hình bằng tham số `key=value`:

    | Mục đích | Lệnh |
    |---|---|
    | Điểm phần LLM (transcript BTC, không lỗi ASR) | `python infer.py subset=val_unseen text=gold` |
    | Điểm cả pipeline có ASR | `python infer.py subset=val_unseen text=asr` |
    | File nộp public test | `python infer.py subset=public_test text=asr` |
    | Chạy thử nhanh | thêm `limit=40` |

    Kết quả (submission, trace từng mẫu, điểm) được đẩy lên `data_repo` (private) ở `outputs/runs/<tên run>/`.
    Dữ liệu gốc tải dạng file tar (`packs/`, tạo bằng `upload_hf --with-raw`) để không vượt giới hạn request của HF.
    Transcript ASR được cache ở `outputs/asr/` nên lần sau không phải chạy lại ASR.
    """
    )
    return


@app.cell
def _():
    from sys import argv as _argv

    CFG = dict(
        model_id="Qwen/Qwen3-4B-Instruct-2507",
        adapter_repo="Jupiter1711/lifemate-qwen3-4b-lora",  # "" = model gốc, không LoRA
        data_repo="Jupiter1711/zalo-lifemate-data",  # HF dataset private: raw/, src/, splits/ (+ outputs/ do notebook này ghi)
        subset="val_unseen",  # val_unseen | val_seen | public_test
        text="gold",  # gold: transcript BTC (chỉ có ở train) | asr: chạy ASR
        asr_model="Jupiter1711/lifemate-phowhisper-large-ft",  # PhoWhisper-large fine-tune (train_asr.py); gốc: vinai/PhoWhisper-large
        asr_bs=32,
        limit=0,  # 0 = tất cả mẫu của subset
        gen_bs=16,  # số prompt mỗi lượt generate
        gen_tok_budget=160_000,  # giới hạn tổng token (padding) một batch generate
        max_new_tokens=384,
        max_steps=10,
        max_invalid=2,
        push_outputs=True,
        work_dir="/tmp/lifemate",
    )
    # Ghi đè bằng tham số dòng lệnh: python infer.py subset=public_test text=asr limit=50
    for _arg in _argv[1:]:
        if "=" not in _arg or _arg.startswith("-"):
            continue
        _k, _v = _arg.split("=", 1)
        if _k not in CFG:
            raise SystemExit(f"tham số lạ: {_k}. Có: {', '.join(CFG)}")
        _d = CFG[_k]
        CFG[_k] = (_v.lower() in ("1", "true", "yes")) if isinstance(_d, bool) else type(_d)(_v) if isinstance(_d, int) else _v
    assert CFG["subset"] in ("val_unseen", "val_seen", "public_test"), CFG["subset"]
    assert CFG["text"] in ("gold", "asr"), CFG["text"]
    if CFG["subset"] == "public_test" and CFG["text"] == "gold":
        raise SystemExit("public_test không có transcript: dùng text=asr")
    print("CFG:", {k: CFG[k] for k in ("subset", "text", "limit", "adapter_repo", "asr_model")})
    return (CFG,)


@app.cell
def _():
    import os as _os
    import subprocess as _sp
    import sys as _sys

    # molab cài sẵn TensorFlow + Keras 3: tắt TF trước khi import transformers (lỗi "Keras 3 not supported").
    _os.environ["USE_TF"] = "0"
    _os.environ["USE_TORCH"] = "1"
    _os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    _pkgs = ["transformers==4.57.1", "peft==0.17.1", "accelerate==1.10.1", "huggingface_hub>=0.34,<1.0"]
    _sp.run([_sys.executable, "-m", "pip", "install", "-q", *_pkgs], check=True)

    import torch
    import transformers
    import peft

    print("torch", torch.__version__, "| transformers", transformers.__version__, "| peft", peft.__version__)
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
    run_btn = mo.ui.run_button(label="Chạy")
    run_btn
    return (run_btn,)


@app.cell
def _(CFG, HF_TOKEN, Path, json, mo, run_btn, time, torch):
    # Mở trong editor thì chờ bấm nút; chạy bằng `python infer.py` thì chạy luôn. Các ô sau đều phụ thuộc ô này.
    mo.stop(not (run_btn.value or mo.app_meta().mode == "script"), mo.md("Bấm **Chạy** để bắt đầu."))
    import sys
    import tarfile
    import warnings

    from huggingface_hub import hf_hub_download as _hf_dl
    from huggingface_hub import snapshot_download

    warnings.simplefilter("ignore", ResourceWarning)  # zaic_tools mở state.json không đóng
    assert torch is not None
    SPLIT = "public_test" if CFG["subset"] == "public_test" else "train"
    _t = time.time()
    # code + splits: vài chục file nhỏ
    DATA_DIR = Path(snapshot_download(CFG["data_repo"], repo_type="dataset", token=HF_TOKEN,
                                      local_dir=f"{CFG['work_dir']}/data", allow_patterns=["src/*", "splits/*"]))
    # dữ liệu gốc: tải file tar trong packs/ (src/tools/upload_hf.py --with-raw) rồi giải nén.
    # Không tải từng file lẻ: hàng nghìn state.json/wav vượt giới hạn 1000 request / 5 phút của HF (lỗi 429).
    RAW = Path(CFG["work_dir"]) / "raw"
    for _name in [f"{SPLIT}_core.tar.gz"] + ([f"{SPLIT}_audio.tar"] if CFG["text"] == "asr" else []):
        _tar = Path(_hf_dl(CFG["data_repo"], f"packs/{_name}", repo_type="dataset", token=HF_TOKEN,
                                    local_dir=f"{CFG['work_dir']}/dl"))
        _done = RAW / f".{_name}.done"
        _stamp = f"{_tar.stat().st_size}:{_tar.stat().st_mtime_ns}"
        if not (_done.exists() and _done.read_text() == _stamp):  # pack mới hoặc chưa giải nén
            with tarfile.open(_tar) as _tf:
                _tf.extractall(RAW, filter="data")
            _done.write_text(_stamp)
        print(f"{_name}: {_tar.stat().st_size / 2**20:.0f} MB, đã giải nén vào {RAW}")
    if str(DATA_DIR) not in sys.path:
        sys.path.insert(0, str(DATA_DIR))  # để import src.* (code đã upload cùng dữ liệu)

    from src.data.common import load_jsonl

    _all = {s["id"]: s for s in load_jsonl(RAW / SPLIT / f"{SPLIT}.jsonl")}
    if CFG["subset"] == "public_test":
        _ids = list(_all)
    else:
        _ids = json.load(open(DATA_DIR / "splits" / "splits.json", encoding="utf-8"))[CFG["subset"]]
    if CFG["limit"]:
        _ids = _ids[:: max(1, len(_ids) // CFG["limit"])][: CFG["limit"]]  # lấy rải đều
    samples = [_all[i] for i in _ids]
    print(f"{len(samples)} mẫu ({CFG['subset']}, split {SPLIT}) | tải dữ liệu {time.time() - _t:.0f}s")
    return RAW, SPLIT, samples


@app.cell
def _(CFG, HF_TOKEN, Path, RAW, SPLIT, json, samples, time, torch):
    # ---------------- ASR (chỉ khi text=asr). Cache: {đường dẫn audio: transcript thô}
    import wave

    import numpy as np
    from huggingface_hub import HfApi, hf_hub_download

    from src.text.vi_numbers import asr_target, llm_input

    # cache theo tên + commit của model ASR: train lại ASR vào cùng repo thì cache cũ không bị dùng nhầm
    _sha = HfApi(token=HF_TOKEN).model_info(CFG["asr_model"]).sha[:8] if CFG["text"] == "asr" else "na"
    _short = f'{CFG["asr_model"].split("/")[-1]}@{_sha}'

    _cache_rel = f"outputs/asr/{_short}/{SPLIT}.json"
    _cache_local = Path(CFG["work_dir"]) / _cache_rel
    transcripts = {}
    if CFG["text"] == "asr":
        try:
            _p = hf_hub_download(CFG["data_repo"], _cache_rel, repo_type="dataset", token=HF_TOKEN,
                                 local_dir=CFG["work_dir"])
            transcripts = json.load(open(_p, encoding="utf-8"))
            print(f"cache ASR: {len(transcripts)} transcript từ {_cache_rel}")
        except Exception:
            print("chưa có cache ASR, chạy ASR")

        _need = []
        for _s in samples:
            for _a in [t["audio"] for t in _s.get("history") or [] if t["role"] == "user"] + [_s["audio"]]:
                if _a not in transcripts and _a not in _need:
                    _need.append(_a)
        if _need:
            from transformers import WhisperForConditionalGeneration, WhisperProcessor

            def _read(rel):
                with wave.open(str(RAW / SPLIT / rel)) as w:
                    assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (16000, 1, 2), rel
                    return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768

            _t0 = time.time()
            _proc = WhisperProcessor.from_pretrained(CFG["asr_model"], token=HF_TOKEN)  # token: model fine-tune ở repo private
            _asr = WhisperForConditionalGeneration.from_pretrained(CFG["asr_model"], dtype=torch.float16,
                                                                   token=HF_TOKEN).to("cuda").eval()
            _audio = {a: _read(a) for a in _need}
            _order = sorted(_need, key=lambda a: len(_audio[a]))
            _lang = {"language": "vi", "task": "transcribe"}
            for _i in range(0, len(_order), CFG["asr_bs"]):
                _b = _order[_i: _i + CFG["asr_bs"]]
                _feats = _proc([_audio[a] for a in _b], sampling_rate=16000, return_tensors="pt").input_features
                _feats = _feats.to("cuda", torch.float16)
                with torch.no_grad():
                    try:
                        _out = _asr.generate(_feats, max_new_tokens=200, **_lang)
                    except ValueError as _e:  # generation_config không có bảng ngôn ngữ -> để model tự nhận
                        print("không ép được language=vi:", str(_e)[:120])
                        _lang = {}
                        _out = _asr.generate(_feats, max_new_tokens=200)
                for _a, _txt in zip(_b, _proc.batch_decode(_out, skip_special_tokens=True)):
                    transcripts[_a] = _txt.strip()
                if (_i // CFG["asr_bs"]) % 10 == 0:
                    print(f"ASR {min(_i + CFG['asr_bs'], len(_order))}/{len(_order)}", flush=True)
            print(f"ASR xong {len(_need)} audio trong {time.time() - _t0:.0f}s")
            del _asr
            torch.cuda.empty_cache()
            _cache_local.parent.mkdir(parents=True, exist_ok=True)
            _cache_local.write_text(json.dumps(transcripts, ensure_ascii=False, indent=0), encoding="utf-8")
            if CFG["push_outputs"]:
                HfApi(token=HF_TOKEN).upload_file(path_or_fileobj=str(_cache_local), path_in_repo=_cache_rel,
                                                  repo_id=CFG["data_repo"], repo_type="dataset",
                                                  commit_message=f"ASR cache {_short} {SPLIT}")

    # Text user đưa vào LLM: cùng đường với dữ liệu SFT (asr_target -> llm_input)
    if CFG["text"] == "gold":
        def user_text(turn):
            return llm_input(asr_target(turn["audio_text"]))
    else:
        def user_text(turn):
            return llm_input(asr_target(transcripts.get(turn["audio"], "")))

    # WER của ASR trên lượt cuối (chỉ có khi có transcript gold)
    def _wer_pair(ref, hyp):
        r, h = ref.split(), hyp.split()
        d = list(range(len(h) + 1))
        for i in range(1, len(r) + 1):
            prev, d[0] = d[0], i
            for j in range(1, len(h) + 1):
                prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
        return d[len(h)], len(r)

    if CFG["text"] == "asr" and samples and samples[0].get("audio_text"):
        _e = _n = 0
        for _s in samples:
            _a, _b = _wer_pair(asr_target(_s["audio_text"]), asr_target(transcripts[_s["audio"]]))
            _e, _n = _e + _a, _n + _b
        print(f"WER lượt cuối ({CFG['asr_model']}): {_e / max(_n, 1):.2%} trên {len(samples)} mẫu")
    _s0 = samples[0]
    print("ví dụ text vào LLM:", user_text({"audio": _s0["audio"], "audio_text": _s0.get("audio_text")}))
    return HfApi, transcripts, user_text


@app.cell
def _(BF16, CFG, HF_TOKEN, Path, samples, torch):
    from huggingface_hub import snapshot_download as _snap
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    assert samples  # chờ ô dữ liệu (nút Chạy) rồi mới nạp model
    tokenizer = AutoTokenizer.from_pretrained(CFG["model_id"])
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        CFG["model_id"], dtype=torch.bfloat16 if BF16 else torch.float16, attn_implementation="sdpa", device_map={"": 0})
    if CFG["adapter_repo"]:
        _dir = Path(_snap(CFG["adapter_repo"], token=HF_TOKEN, local_dir=f"{CFG['work_dir']}/adapter_dl",
                          allow_patterns=["adapter_*", "last-checkpoint/adapter_*"]))
        _src = _dir if (_dir / "adapter_config.json").exists() else _dir / "last-checkpoint"
        print("LoRA từ", CFG["adapter_repo"], "/", _src.relative_to(_dir) if _src != _dir else "(gốc repo)")
        model = PeftModel.from_pretrained(model, str(_src)).merge_and_unload()
    else:
        print("chạy model gốc (không LoRA)")
    model.eval()
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    return model, tokenizer


@app.cell
def _(CFG, model, tokenizer, torch):
    def policy(episodes):
        """Sinh một lượt cho mỗi episode (greedy), gom batch theo độ dài prompt để ít padding."""
        prompts = [tokenizer.apply_chat_template(ep.messages, tools=ep.tools, tokenize=False, add_generation_prompt=True)
                   for ep in episodes]
        enc_all = [tokenizer(p, add_special_tokens=False)["input_ids"] for p in prompts]
        order = sorted(range(len(prompts)), key=lambda i: len(enc_all[i]))
        outs = [""] * len(prompts)
        i = 0
        while i < len(order):
            j = i + 1  # thêm prompt vào batch đến khi đủ gen_bs hoặc vượt ngân sách token
            while (j < len(order) and j - i < CFG["gen_bs"]
                   and (j - i + 1) * (len(enc_all[order[j]]) + CFG["max_new_tokens"]) <= CFG["gen_tok_budget"]):
                j += 1
            batch = order[i:j]
            enc = tokenizer.pad({"input_ids": [enc_all[k] for k in batch]}, return_tensors="pt").to(model.device)
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=CFG["max_new_tokens"], do_sample=False,
                                     pad_token_id=tokenizer.pad_token_id)
            texts = tokenizer.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            for k, t in zip(batch, texts):
                outs[k] = t.strip()
            i = j
        return outs

    return (policy,)


@app.cell
def _(CFG, Path, RAW, SPLIT, json, policy, samples, time, transcripts, user_text):
    import collections

    from src.agent.loop import AgentRunner, submission_row

    run_name = f"{CFG['subset']}_{CFG['text']}_{'lora' if CFG['adapter_repo'] else 'base'}_{time.strftime('%m%d-%H%M')}"
    out_dir = Path(CFG["work_dir"]) / "runs" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    _t0 = time.time()
    runner = AgentRunner(RAW, SPLIT, user_text, max_steps=CFG["max_steps"], max_invalid=CFG["max_invalid"])
    episodes = runner.run(samples, policy, log=lambda m: print(f"{m} | {time.time() - _t0:.0f}s", flush=True))
    print(f"agent xong {len(episodes)} mẫu trong {(time.time() - _t0) / 60:.1f} phút")
    print("lý do dừng:", dict(collections.Counter(e.stop_reason for e in episodes)))

    rows = [submission_row(e) for e in episodes]
    with open(out_dir / "submission.jsonl", "w", encoding="utf-8") as _f:
        for _r in rows:
            _f.write(json.dumps(_r, ensure_ascii=False) + "\n")
    with open(out_dir / "trace.jsonl", "w", encoding="utf-8") as _f:
        for _e in episodes:
            _f.write(json.dumps({"id": _e.id,
                                 "user_text": user_text({"audio": _e.sample["audio"], "audio_text": _e.sample.get("audio_text")}),
                                 "asr_raw": transcripts.get(_e.sample["audio"]), "stop": _e.stop_reason,
                                 "calls": _e.calls, "trace": _e.trace}, ensure_ascii=False) + "\n")
    print(f"-> {out_dir}/submission.jsonl ({len(rows)} dòng)")
    return episodes, out_dir, rows, run_name


@app.cell
def _(CFG, HF_TOKEN, HfApi, RAW, SPLIT, episodes, json, out_dir, rows, run_name):
    # Chấm điểm (chỉ val có gold) rồi đẩy kết quả lên HF private
    if SPLIT == "train":
        from src.eval.scorer import Scorer, print_summary, summarize

        scorer = Scorer(RAW, "train")
        details = [scorer.score_sample(r["id"], r["tool_calls"], r["state_hash"]) for r in rows]
        summary = summarize(details)
        print_summary(summary, 0)
        (out_dir / "score.json").write_text(json.dumps({"cfg": CFG, "summary": summary}, ensure_ascii=False, indent=1),
                                            encoding="utf-8")
        with open(out_dir / "details.jsonl", "w", encoding="utf-8") as _f:
            for _d in details:
                _f.write(json.dumps(_d, ensure_ascii=False) + "\n")
        # vài mẫu 0 điểm để xem lỗi
        _ep = {e.id: e for e in episodes}
        _shown = 0
        for _d in details:
            if _d["score"] > 0 or _shown >= 6:
                continue
            _shown += 1
            _s = scorer.samples[_d["id"]]
            print(f"\n[{_d['type']}] {_d['id']} | hash {'đúng' if _d['hash_ok'] else 'sai'} | dừng: {_ep[_d['id']].stop_reason}")
            print("  user:", _s["audio_text"])
            print("  pred:", json.dumps(_ep[_d["id"]].calls, ensure_ascii=False)[:600])
            print("  gold:", json.dumps(_s["label"]["tool_calls"], ensure_ascii=False)[:600])
    else:
        assert len(rows) == len({r["id"] for r in rows}), "trùng id"
        print(f"public_test: {len(rows)} dòng, {sum(bool(r['tool_calls']) for r in rows)} dòng có tool call")

    if CFG["push_outputs"]:
        HfApi(token=HF_TOKEN).upload_folder(folder_path=str(out_dir), path_in_repo=f"outputs/runs/{run_name}",
                                            repo_id=CFG["data_repo"], repo_type="dataset", commit_message=f"run {run_name}")
        print(f"đã đẩy lên https://huggingface.co/datasets/{CFG['data_repo']}/tree/main/outputs/runs/{run_name}")
    return


if __name__ == "__main__":
    app.run()
