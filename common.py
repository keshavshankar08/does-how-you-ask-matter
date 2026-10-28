"""Shared helpers: GPU pinning, model loading, sharding, resumable job runner.

Env vars read here:
  ALLOW   which GPU this process may use, in nvidia-smi numbering, e.g. "3"
  SHARD   "i/n", so n processes split the rows between them
  MODEL   hugging face model id
"""

import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
LOGS = ROOT / "logs"

NO_THINK = os.getenv("NO_THINK", "1") == "1"
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def device():
    """Pin to the GPU in ALLOW. Must run before torch sees any device."""
    allow = os.getenv("ALLOW", "").strip()
    if allow:
        # PCI_BUS_ID makes CUDA number devices the way nvidia-smi does. Without it
        # CUDA orders by capability and ALLOW=0 can mean a different card.
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = allow
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def shard():
    index, total = (int(v) for v in os.getenv("SHARD", "0/1").split("/"))
    return index, total


def shard_tag():
    index, total = shard()
    return f"s{index}of{total}"


def my_rows(rows):
    """This process's share. Shards are disjoint, so every row is done once."""
    index, total = shard()
    return rows[index::total] if total > 1 else list(rows)


def shard_path(base):
    """results/x/name.jsonl becomes results/x/name.s0of4.jsonl when sharded."""
    base = Path(base)
    index, total = shard()
    if total == 1:
        return base
    return base.with_suffix(f".{shard_tag()}{base.suffix}")


def load_shards(base):
    """Every shard of a stage output, plus the unsharded file if it exists."""
    base = Path(base)
    rows = []
    if base.exists():
        rows += load_jsonl(base)
    for path in sorted(base.parent.glob(f"{base.stem}.s*{base.suffix}")):
        rows += load_jsonl(path)
    return rows


def setup_logging(stage):
    LOGS.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = LOGS / f"{stage}_{shard_tag()}_{stamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(path), logging.StreamHandler(sys.stdout)],
        force=True,
    )
    for noisy in ("httpx", "httpcore", "urllib3", "huggingface_hub", "filelock",
                  "transformers", "accelerate"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    log = logging.getLogger(stage)
    log.info("log file: %s", path)
    return log


class Model:
    """Loaded on first use, so a stage that makes no model call loads nothing."""

    def __init__(self, name=None):
        self.name = name or os.getenv("MODEL", "Qwen/Qwen3-32B")
        self.pipe = None

    def load(self):
        if self.pipe is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        where = device()       # sets CUDA_VISIBLE_DEVICES from ALLOW before torch sees it
        print(f"[model] {self.name} on {where}", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.pipe = AutoModelForCausalLM.from_pretrained(
            self.name, dtype=torch.bfloat16,
            device_map="auto" if where == "cuda" else None)
        if where != "cuda":
            self.pipe = self.pipe.to(where)
        self.pipe.eval()

    def chat(self, messages, max_new=256, temperature=0.0):
        """One reply. temperature 0 means greedy."""
        self.load()
        import torch
        if NO_THINK:
            # /no_think works on Qwen3 whatever the transformers version does with the
            # enable_thinking flag, so both are set rather than relying on either
            messages = list(messages)
            messages[-1] = {**messages[-1],
                            "content": messages[-1]["content"] + "\n/no_think"}
        try:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.pipe.device)
        with torch.no_grad():
            out = self.pipe.generate(
                **inputs, max_new_tokens=max_new,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                top_p=0.8 if temperature > 0 else None,
                pad_token_id=self.tokenizer.pad_token_id)
        reply = self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                      skip_special_tokens=True)
        return THINK_RE.sub("", reply).strip()


def load_jsonl(path):
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def done_ids(path, id_field):
    path = Path(path)
    if not path.exists():
        return set()
    return {json.loads(line)[id_field] for line in open(path, encoding="utf-8") if line.strip()}


def run_jobs(jobs, worker, output_path, id_field, log, label="jobs"):
    """Run worker(job) over this shard's jobs, writing each result as it finishes.

    Ids already in the output file are skipped, so rerunning the same command after
    an interruption continues where it stopped.
    """
    output_path = shard_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mine = my_rows(jobs)
    done = done_ids(output_path, id_field)
    todo = [job for job in mine if job[id_field] not in done]
    log.info("%s: %d in this shard, %d already done, %d to run", label, len(mine), len(done), len(todo))
    if not todo:
        return 0

    ok = failed = 0
    started = time.time()
    with output_path.open("a", encoding="utf-8") as stream:
        for job in todo:
            try:
                result = worker(job)
            except Exception as error:
                failed += 1
                log.error("%s failed: %s", job[id_field], error)
                continue
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            stream.flush()
            ok += 1
            if ok % 25 == 0 or ok + failed == len(todo):
                rate = (ok + failed) / max(time.time() - started, 1e-6)
                left = (len(todo) - ok - failed) / max(rate, 1e-6)
                log.info("%d/%d done, %.2f/s, ~%.0f min left, %d failed",
                         ok + failed, len(todo), rate, left / 60, failed)

    log.info("%s: wrote %d, failed %d, to %s", label, ok, failed, output_path)
    return failed


ANSWER_RE = re.compile(r"answer\s*[:\-]?\s*\(?([A-Z])\)?", re.IGNORECASE)


def parse_answer(text, valid_letters):
    """The chosen option letter, or None if the reply does not give one."""
    if not text:
        return None
    match = ANSWER_RE.search(text)
    if match and match.group(1).upper() in valid_letters:
        return match.group(1).upper()
    for line in text.splitlines():
        bare = line.strip().strip("().*: ").upper()
        if bare in valid_letters:
            return bare
    return None


def options_block(options):
    return "\n".join(f"({letter}) {text}" for letter, text in sorted(options.items()))