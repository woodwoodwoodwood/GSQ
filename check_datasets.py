"""Standalone dataset-loadability checker for lm-eval tasks.

This does NOT require a model / vLLM server. It only exercises lm-eval's
dataset loading path so you can confirm whether a benchmark's data can be
READ by your installed lm_eval (online or from a local path) before you
bother deploying a model.

For each task it reports:
  [OK]   task registered + dataset loaded + a sample doc fetched
  [FAIL] not registered, or dataset failed to load (with the error)

Usage (run with your venv python):
  /usr/local/app/GSQ/.venv/bin/python check_datasets.py \
      --tasks mmlu_pro,humaneval,supergpqa,ceval-valid

  # Force offline (fail loudly instead of silently downloading):
  /usr/local/app/GSQ/.venv/bin/python check_datasets.py \
      --tasks mmlu_pro --offline

  # Use custom/override task YAMLs (e.g. local-dataset versions):
  /usr/local/app/GSQ/.venv/bin/python check_datasets.py \
      --tasks mmlu_pro_local --include-path custom_tasks/local

Notes:
  - A "group" task (e.g. ceval-valid has 52 subtasks) will instantiate ALL
    its subtasks, which can download many subsets. Use --max-subtasks to cap
    how many subtasks are actually sample-checked, or pass a single leaf task
    name (e.g. ceval-valid_logic) for a fast check.
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys
import traceback


def parse_args():
    p = argparse.ArgumentParser(__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tasks", required=True,
                   help="Comma-separated lm-eval task names to check.")
    p.add_argument("--include-path", default=None,
                   help="Extra dir with custom/override task YAMLs (lm-eval --include_path).")
    p.add_argument("--offline", action="store_true",
                   help="Set HF_HUB_OFFLINE / HF_DATASETS_OFFLINE so loading fails loudly "
                        "instead of silently downloading. Use to verify LOCAL data works.")
    p.add_argument("--max-subtasks", type=int, default=3,
                   help="For group tasks, how many leaf subtasks to sample-check (default 3).")
    return p.parse_args()


def _iter_leaf_tasks(obj, prefix=""):
    """Flatten lm-eval's load_task_or_group() return into (name, task_obj) pairs."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            name = k if isinstance(k, str) else str(k)
            yield from _iter_leaf_tasks(v, name)
    else:
        yield prefix, obj


def _fetch_one_doc(task_obj):
    """Return (split_used, sample_keys) for the first available doc."""
    for split_name, has, getter in (
        ("test", task_obj.has_test_docs, task_obj.test_docs),
        ("validation", task_obj.has_validation_docs, task_obj.validation_docs),
        ("train", task_obj.has_training_docs, task_obj.training_docs),
    ):
        try:
            if has():
                doc = next(iter(getter()))
                keys = sorted(doc.keys()) if isinstance(doc, dict) else type(doc).__name__
                return split_name, keys
        except StopIteration:
            return split_name, "(empty split)"
        except Exception as e:  # split declared but failed to materialize
            return split_name, f"(error fetching doc: {e})"
    return None, "(no test/validation/train split)"


def main():
    args = parse_args()

    # Must set offline env BEFORE importing datasets/lm_eval.
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        print("[env] HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 (offline mode)")

    try:
        import inspect
        from lm_eval.tasks import TaskManager
    except Exception as e:
        print(f"FATAL: cannot import lm_eval: {e}", file=sys.stderr)
        sys.exit(2)

    # TaskManager's signature varies across lm_eval versions. Some accept
    # `include_path`, some `include_defaults`, some neither. Detect and pass
    # include_path only if supported; otherwise fall back gracefully.
    tm_kwargs = {}
    if args.include_path:
        try:
            sig_params = inspect.signature(TaskManager.__init__).parameters
        except (TypeError, ValueError):
            sig_params = {}
        if "include_path" in sig_params:
            tm_kwargs["include_path"] = args.include_path
        else:
            print(f"[warn] this lm_eval's TaskManager has no 'include_path' kwarg; "
                  f"trying to register include dir another way.")
    try:
        tm = TaskManager(**tm_kwargs)
    except TypeError as e:
        # Last-resort: instantiate with no kwargs.
        print(f"[warn] TaskManager(**{tm_kwargs}) failed ({e}); retrying with no args.")
        tm = TaskManager()

    # If include_path was requested but unsupported as a kwarg, try the
    # module-level task registration API used by some versions.
    if args.include_path and "include_path" not in tm_kwargs:
        try:
            from lm_eval.tasks import include_path as _include_path  # type: ignore
            _include_path(args.include_path)
            tm = TaskManager()  # refresh registry
            print(f"[info] registered include dir via lm_eval.tasks.include_path().")
        except Exception as e:
            print(f"[warn] could not register include dir '{args.include_path}': {e}\n"
                  f"       custom/override tasks under it may not be found.")

    all_tasks = set(tm.all_tasks)

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    print(f"\nChecking {len(tasks)} task(s). lm_eval knows {len(all_tasks)} tasks total.\n")

    results = {}
    for task in tasks:
        print("=" * 70)
        print(f"TASK: {task}")

        if task not in all_tasks:
            print(f"  [FAIL] NOT REGISTERED in this lm_eval install.")
            print(f"         -> cannot be read as-is; needs a custom task YAML "
                  f"(pass --include-path pointing to it).")
            results[task] = "NOT_REGISTERED"
            continue

        try:
            loaded = tm.load_task_or_group(task)
        except Exception as e:
            print(f"  [FAIL] load_task_or_group raised:\n{traceback.format_exc()}")
            results[task] = f"LOAD_ERROR: {e}"
            continue

        leaves = list(_iter_leaf_tasks(loaded))
        n = len(leaves)
        is_group = n > 1
        if is_group:
            print(f"  (group: {n} subtasks; sample-checking up to {args.max_subtasks})")

        ok = 0
        checked = leaves[: args.max_subtasks] if is_group else leaves
        for name, tobj in checked:
            try:
                split, keys = _fetch_one_doc(tobj)
                if split is None:
                    print(f"  [WARN] {name}: {keys}")
                else:
                    print(f"  [OK]   {name}: read from '{split}' split; doc keys={keys}")
                    ok += 1
            except Exception as e:
                print(f"  [FAIL] {name}: {e}")
        results[task] = f"OK ({ok}/{len(checked)} checked)" if ok else "NO_DOCS"

    print("\n" + "=" * 70)
    print("SUMMARY")
    for task, status in results.items():
        print(f"  {task:20s} -> {status}")
    # Non-zero exit if anything is unreadable.
    bad = [t for t, s in results.items() if not str(s).startswith("OK")]
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
