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
    # LifeMate: train LLM (LoRA SFT)

    Fine-tune **Qwen3-4B-Instruct-2507** bằng LoRA trên dữ liệu SFT multi-turn (`src/data/build_sft.py`).
    Chỉ học phần trả lời cho lệnh cuối: các lượt tool call và câu kết thúc; prompt, history và kết quả tool bị mask.

    **Chuẩn bị (một lần):**

    1. Ở máy local: `python -m src.tools.upload_hf --repo <hf_user>/zalo-lifemate-data`, đẩy dữ liệu lên một HF dataset **private**.
    2. Trên molab: tạo file `.env` trong file browser, ghi `HF_TOKEN=hf_...` (token quyền write). File `.env` được giữ
       giữa các phiên và **không** bị copy khi người khác fork notebook.
    3. Sửa `data_repo` và `out_repo` trong ô cấu hình bên dưới.
    4. Bật GPU ở nút specs của notebook.

    **Không upload dữ liệu cuộc thi qua file browser của molab.** Notebook molab mặc định là public; ai có link đều fork được,
    kèm theo các file đã upload.

    Checkpoint được đẩy lên `out_repo` (private) mỗi `save_steps` bước, vì molab dừng phiên sau 12 giờ, và file lớn hơn 1GB
    trong workspace không được giữ lại. Nếu phiên bị dừng, đặt `resume=True` rồi chạy lại.
    """
    )
    return


@app.cell
def _():
    from sys import argv as _argv

    CFG = dict(
        model_id="Qwen/Qwen3-4B-Instruct-2507",
        data_repo="Jupiter1711/zalo-lifemate-data",  # HF dataset private (src/tools/upload_hf.py)
        # Mặc định = cấu hình v2 (text sạch + text ASR k-fold). Chạy lại v1 (public 52.32, tag v0.1-public-52.32):
        #   sft_dir=sft train_splits=train epochs=3 eval_text_src= eval_before=true out_repo=Jupiter1711/lifemate-qwen3-4b-lora
        out_repo="Jupiter1711/lifemate-qwen3-4b-lora-v2",  # HF model repo private để lưu adapter/checkpoint; "" = không push
        sft_dir="sft_v2",  # sft: chỉ text sạch (v1) | sft_v2: mỗi sample 2 bản, text sạch + text ASR k-fold
        train_splits="train,val_seen",  # bản nộp cuối: train,val_seen,val_unseen
        eval_split="val_unseen",
        eval_text_src="asr",  # đo trên bản text ASR (giống lúc chạy thật); "" = mọi bản (bắt buộc "" với sft v1)
        eval_n=80,  # số sample để đo nhanh bước đầu tiên (trước/sau train)
        eval_before=False,  # True: đo model gốc trước khi train (baseline đã biết: 0.45 đúng cả tham số)
        max_len=12288,
        lora_r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        lr=1e-4,
        epochs=2,  # sft_v2 đã nhân đôi dữ liệu (v1: 3)
        grad_accum=8,
        warmup_ratio=0.03,
        save_steps=50,
        seed=2026,
        resume=False,  # True: tiếp tục từ last-checkpoint trên out_repo
        work_dir="/tmp/lifemate",  # ngoài workspace: không bị lưu/fork
    )
    # Ghi đè bằng tham số dòng lệnh: python train_llm.py epochs=3 sft_dir=sft
    for _arg in _argv[1:]:
        if "=" not in _arg or _arg.startswith("-"):
            continue
        _k, _v = _arg.split("=", 1)
        if _k not in CFG:
            raise SystemExit(f"tham số lạ: {_k}. Có: {', '.join(CFG)}")
        _d = CFG[_k]
        CFG[_k] = ((_v.lower() in ("1", "true", "yes")) if isinstance(_d, bool) else type(_d)(_v)
                   if isinstance(_d, (int, float)) else _v)
    print("CFG:", CFG)
    return (CFG,)


@app.cell
def _():
    # Cài thư viện (pin version để chạy lại cho giống nhau). torch dùng bản có sẵn của molab (hỗ trợ GPU Blackwell).
    import os as _os
    import subprocess as _sp
    import sys as _sys

    # molab cài sẵn TensorFlow + Keras 3: transformers sẽ nạp phần TF và lỗi "Keras 3 not supported".
    # Notebook chỉ dùng PyTorch, nên tắt TF trước khi import transformers.
    _os.environ["USE_TF"] = "0"
    _os.environ["USE_TORCH"] = "1"
    _os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    _pkgs = ["transformers==4.57.1", "peft==0.17.1", "accelerate==1.10.1", "huggingface_hub>=0.34,<1.0"]
    _sp.run([_sys.executable, "-m", "pip", "install", "-q", *_pkgs], check=True)

    import torch
    import transformers
    import peft

    print("torch", torch.__version__, "| transformers", transformers.__version__, "| peft", peft.__version__)
    deps_ok = True
    return deps_ok, torch


@app.cell
def _(deps_ok, torch):
    assert deps_ok and torch.cuda.is_available(), "Chưa có GPU: bật GPU ở nút specs của notebook"
    _p = torch.cuda.get_device_properties(0)
    BF16 = torch.cuda.is_bf16_supported()
    print(f"GPU: {_p.name} | VRAM {_p.total_memory / 2**30:.0f} GB | bf16: {BF16}")
    if _p.total_memory / 2**30 < 40:
        print("VRAM < 40GB (vd. T4 trên Kaggle): giảm max_len xuống ~8192, lora_r=16, hoặc chuyển sang QLoRA 4-bit.")
    return (BF16,)


@app.cell
def _():
    import json
    import os
    from pathlib import Path

    # Đọc .env (molab giữ file này giữa các phiên)
    _env = Path(".env")
    if _env.exists():
        for _line in _env.read_text(encoding="utf-8").splitlines():
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip().strip("'\""))
    HF_TOKEN = os.environ.get("HF_TOKEN")
    assert HF_TOKEN, "Thiếu HF_TOKEN: tạo file .env với dòng HF_TOKEN=hf_..."
    return HF_TOKEN, Path, json


@app.cell
def _(CFG, HF_TOKEN, Path, deps_ok, json):
    from huggingface_hub import snapshot_download

    assert deps_ok
    DATA_DIR = Path(snapshot_download(CFG["data_repo"], repo_type="dataset", token=HF_TOKEN,
                                      local_dir=f"{CFG['work_dir']}/data",
                                      allow_patterns=[f"{CFG['sft_dir']}/*", "splits/*"]))

    def _load(name):
        with open(DATA_DIR / CFG["sft_dir"] / f"{name}.jsonl", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    train_recs = [r for _s in CFG["train_splits"].split(",") for r in _load(_s.strip())]
    eval_recs = [r for r in _load(CFG["eval_split"])
                 if not CFG["eval_text_src"] or r.get("text_src", "clean") == CFG["eval_text_src"]]
    print(f"train: {len(train_recs)} mẫu từ {CFG['train_splits']} | eval: {len(eval_recs)} mẫu ({CFG['eval_split']})")
    return eval_recs, snapshot_download, train_recs


@app.cell
def _(CFG, deps_ok):
    from transformers import AutoTokenizer

    assert deps_ok
    tokenizer = AutoTokenizer.from_pretrained(CFG["model_id"])

    def render(messages, tools, gen=False):
        return tokenizer.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=gen)

    def encode(rec):
        """-> ({input_ids, labels}, None) hoặc (None, lý do bỏ). Chỉ học các message assistant từ loss_from trở đi."""
        assist_header = "<|im_start|>assistant\n"  # không học header, học nội dung + <|im_end|>
        msgs, tools, start = rec["messages"], rec["tools"], rec["loss_from"]
        full = render(msgs, tools)
        prev = render(msgs[:start], tools)
        if not full.startswith(prev):
            return None, "template không giữ prefix"
        spans = []
        for k in range(start, len(msgs)):
            cur = render(msgs[: k + 1], tools)
            if not cur.startswith(prev) or not full.startswith(cur):
                return None, "template không giữ prefix"
            if msgs[k]["role"] == "assistant":
                s = len(prev) + (len(assist_header) if cur.startswith(assist_header, len(prev)) else 0)
                spans.append((s, len(cur)))  # gồm cả <|im_end|> để model học dừng
            prev = cur
        enc = tokenizer(full, return_offsets_mapping=True, add_special_tokens=False)
        ids = enc["input_ids"]
        if len(ids) > CFG["max_len"]:
            return None, "dài quá max_len"
        labels = [t if any(a <= o0 < b for a, b in spans) else -100 for t, (o0, _o1) in zip(ids, enc["offset_mapping"])]
        return {"input_ids": ids, "labels": labels}, None

    return encode, render, tokenizer


@app.cell
def _(encode, tokenizer, train_recs):
    import collections

    train_ds, _drop, _lens = [], collections.Counter(), []
    for _r in train_recs:
        _ex, _why = encode(_r)
        if _ex is None:
            _drop[_why] += 1
            continue
        train_ds.append(_ex)
        _lens.append(len(_ex["input_ids"]))
    _lens.sort()
    _n_lab = sum(sum(t != -100 for t in e["labels"]) for e in train_ds)
    print(f"dùng {len(train_ds)} mẫu, bỏ {dict(_drop)}")
    print(f"token/mẫu: p50 {_lens[len(_lens) // 2]}, p90 {_lens[int(len(_lens) * .9)]}, max {_lens[-1]} | "
          f"tổng {sum(_lens):,} token, trong đó {_n_lab:,} token được học")
    _e = train_ds[0]
    print("\n--- phần được học của mẫu đầu tiên ---\n" + tokenizer.decode([t for t in _e["labels"] if t != -100]))
    return (train_ds,)


@app.cell
def _(BF16, CFG, deps_ok, torch):
    from transformers import AutoModelForCausalLM, set_seed
    from peft import LoraConfig, get_peft_model

    assert deps_ok
    set_seed(CFG["seed"])
    _base = AutoModelForCausalLM.from_pretrained(
        CFG["model_id"], dtype=torch.bfloat16 if BF16 else torch.float16,
        attn_implementation="sdpa", device_map={"": 0},
    )
    _base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    _base.enable_input_require_grads()
    _base.config.use_cache = False
    model = get_peft_model(_base, LoraConfig(
        r=CFG["lora_r"], lora_alpha=CFG["lora_alpha"], lora_dropout=CFG["lora_dropout"], task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    ))
    model.print_trainable_parameters()
    return model, set_seed


@app.cell
def _(json, render, tokenizer, torch):
    import re

    def first_step_eval(model, recs, n, show_wrong=8):
        """Đo nhanh: cho prompt đến lệnh cuối, model sinh lượt đầu tiên, so với gold (tool call đầu tiên, hoặc không gọi
        tool với mẫu ngoài phạm vi). Chỉ là chỉ số tham khảo; điểm thật cần chạy cả vòng agent + scorer."""
        tool_re = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)

        def canon(v):
            return json.dumps(v, ensure_ascii=False, sort_keys=True).lower()

        step = max(1, len(recs) // n)
        picked = recs[::step][:n]
        model.eval()
        model.config.use_cache = True
        hit_name = hit_exact = 0
        by_type = {}
        wrong = []
        for rec in picked:
            prompt = render(rec["messages"][: rec["loss_from"]], rec["tools"], gen=True)
            ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
            with torch.no_grad():
                out = model.generate(**ids, max_new_tokens=300, do_sample=False)
            text = tokenizer.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
            m = tool_re.search(text)
            try:
                pred = json.loads(m.group(1)) if m else None
            except json.JSONDecodeError:
                pred = {"name": "<json lỗi>"}
            gold_msg = rec["messages"][rec["loss_from"]]
            gold = gold_msg["tool_calls"][0]["function"] if gold_msg.get("tool_calls") else None
            ok_name = (pred is None and gold is None) or (pred is not None and gold is not None and pred.get("name") == gold["name"])
            ok_exact = ok_name and (gold is None or canon(pred.get("arguments")) == canon(gold["arguments"]))
            hit_name += ok_name
            hit_exact += ok_exact
            if not ok_exact:
                wrong.append((rec["id"], rec["type"], gold, pred))
            t = by_type.setdefault(rec["type"], [0, 0])
            t[0] += ok_exact
            t[1] += 1
        model.config.use_cache = False
        res = {"n": len(picked), "đúng tên tool": round(hit_name / len(picked), 3),
               "đúng cả tham số": round(hit_exact / len(picked), 3),
               "theo loại (đúng cả tham số)": {k: f"{a}/{b}" for k, (a, b) in sorted(by_type.items())}}
        print(res)
        print(f"-- {min(show_wrong, len(wrong))}/{len(wrong)} mẫu sai đầu tiên (gold | pred):")
        for sid, typ, g, p in wrong[:show_wrong]:
            print(f"  [{typ}] {sid}\n    gold: {json.dumps(g, ensure_ascii=False)}\n    pred: {json.dumps(p, ensure_ascii=False)}")
        return res

    return (first_step_eval,)


@app.cell
def _(mo):
    train_btn = mo.ui.run_button(label="Train")
    train_btn
    return (train_btn,)


@app.cell
def _(
    BF16,
    CFG,
    HF_TOKEN,
    Path,
    eval_recs,
    first_step_eval,
    json,
    mo,
    model,
    set_seed,
    snapshot_download,
    tokenizer,
    torch,
    train_btn,
    train_ds,
):
    mo.stop(not (train_btn.value or mo.app_meta().mode == "script"), mo.md("Bấm **Train** để bắt đầu."))
    from transformers import PrinterCallback, Trainer, TrainerCallback, TrainingArguments

    if CFG["eval_before"] and not CFG["resume"]:
        print("== model gốc (trước train)")
        eval_before = first_step_eval(model, eval_recs, CFG["eval_n"])

    def _collate(batch):
        L = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), L), tokenizer.pad_token_id, dtype=torch.long)
        lab = torch.full((len(batch), L), -100, dtype=torch.long)
        att = torch.zeros((len(batch), L), dtype=torch.long)
        for i, b in enumerate(batch):
            n = len(b["input_ids"])
            ids[i, :n] = torch.tensor(b["input_ids"])
            lab[i, :n] = torch.tensor(b["labels"])
            att[i, :n] = 1
        return {"input_ids": ids, "labels": lab, "attention_mask": att}

    _ckpt_dir = f"{CFG['work_dir']}/ckpt"
    _resume = None
    if CFG["resume"] and CFG["out_repo"]:
        snapshot_download(CFG["out_repo"], token=HF_TOKEN, local_dir=_ckpt_dir, allow_patterns=["last-checkpoint/*"])
        _resume = str(Path(_ckpt_dir) / "last-checkpoint") if (Path(_ckpt_dir) / "last-checkpoint").exists() else None
        print("tiếp tục từ", _resume)

    set_seed(CFG["seed"])
    _args = TrainingArguments(
        output_dir=_ckpt_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=CFG["grad_accum"],
        num_train_epochs=CFG["epochs"],
        learning_rate=CFG["lr"],
        lr_scheduler_type="cosine",
        warmup_ratio=CFG["warmup_ratio"],
        bf16=BF16,
        fp16=not BF16,
        logging_steps=5,
        save_strategy="steps",
        save_steps=CFG["save_steps"],
        save_total_limit=2,
        report_to="none",
        disable_tqdm=True,  # chạy qua subprocess, thanh tiến trình notebook chỉ in ra "<IPython.core.display.HTML object>"
        remove_unused_columns=False,
        seed=CFG["seed"],
        data_seed=CFG["seed"],
        dataloader_num_workers=2,
        push_to_hub=bool(CFG["out_repo"]),
        hub_model_id=CFG["out_repo"] or None,
        hub_private_repo=True,
        hub_strategy="checkpoint",  # đẩy cả last-checkpoint (kèm optimizer) để resume được
        hub_token=HF_TOKEN,
    )
    class _LogCb(TrainerCallback):
        """Một dòng text mỗi logging_steps: bước, epoch, loss, lr, thời gian còn lại."""

        def on_train_begin(self, args, state, control, **kw):
            import time
            self.t0, self.s0 = time.time(), state.global_step

        def on_log(self, args, state, control, logs=None, **kw):
            if not logs or "loss" not in logs:
                return
            import time
            done = state.global_step - self.s0
            spent = time.time() - self.t0
            eta = spent / max(done, 1) * (state.max_steps - state.global_step)
            print(f"step {state.global_step}/{state.max_steps} | epoch {logs.get('epoch', 0):.2f} | "
                  f"loss {logs['loss']:.4f} | grad {logs.get('grad_norm', 0):.3f} | lr {logs.get('learning_rate', 0):.2e} | "
                  f"đã chạy {spent / 60:.0f}' còn ~{eta / 60:.0f}'", flush=True)

    trainer = Trainer(model=model, args=_args, train_dataset=train_ds, data_collator=_collate)
    trainer.remove_callback(PrinterCallback)  # thay bằng _LogCb cho gọn
    trainer.add_callback(_LogCb())
    trainer.train(resume_from_checkpoint=_resume)

    # loss trung bình theo epoch + lưu log (nằm trong ckpt dir nên được push lên HF cùng adapter)
    _hist = [h for h in trainer.state.log_history if "loss" in h]
    _by_ep = {}
    for _h in _hist:
        _by_ep.setdefault(min(int(_h["epoch"] - 1e-9), CFG["epochs"] - 1) + 1, []).append(_h["loss"])
    print("loss trung bình theo epoch:", {e: round(sum(v) / len(v), 4) for e, v in sorted(_by_ep.items())})
    Path(_ckpt_dir).mkdir(parents=True, exist_ok=True)
    (Path(_ckpt_dir) / "train_log.json").write_text(
        json.dumps({"cfg": CFG, "log_history": trainer.state.log_history}, ensure_ascii=False, indent=1), encoding="utf-8")

    _adapter_dir = f"{CFG['work_dir']}/adapter"
    trainer.save_model(_adapter_dir)
    tokenizer.save_pretrained(_adapter_dir)
    if CFG["out_repo"]:
        trainer.push_to_hub(commit_message="final adapter")
        print(f"đã đẩy adapter lên https://huggingface.co/{CFG['out_repo']} (private)")
    return (trainer,)


@app.cell
def _(CFG, eval_recs, first_step_eval, trainer):
    print("== sau train")
    eval_after = first_step_eval(trainer.model, eval_recs, CFG["eval_n"])
    return


if __name__ == "__main__":
    app.run()
