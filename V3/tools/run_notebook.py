"""Execute ``V3/V3_SHD_experiment.ipynb`` top-to-bottom with a real kernel.

The repository virtual environment ships ``ipykernel`` + ``jupyter_client`` but no
``nbformat``/``nbclient``, so this script talks to the kernel directly and writes
the outputs back into the ``.ipynb`` in nbformat 4 layout.

    .venv\\Scripts\\python.exe V3\\tools\\run_notebook.py                 # full run
    $env:V3_QUICK=1; .venv\\Scripts\\python.exe V3\\tools\\run_notebook.py  # smoke run

Options: ``--cells N`` (execute only the first N code cells), ``--timeout S`` per
cell, ``--in-place`` (overwrite the notebook, the default), ``--dry-run``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from queue import Empty

V3_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = V3_ROOT / "V3_SHD_experiment.ipynb"

try:  # tqdm emits box-drawing characters; the Windows console default is cp1252
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass


def _text(v) -> str:
    return "".join(v) if isinstance(v, list) else str(v)


def _console(text: str) -> None:
    try:
        sys.stdout.write(text)
    except UnicodeEncodeError:
        sys.stdout.write(text.encode("ascii", "replace").decode("ascii"))
    sys.stdout.flush()


def execute(notebook: Path, max_cells: int | None, cell_timeout: float, out: Path) -> int:
    from jupyter_client.manager import KernelManager

    nb = json.loads(notebook.read_text(encoding="utf-8"))
    km = KernelManager(kernel_name="python3")
    env = dict(os.environ)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("MPLBACKEND", "Agg")
    km.start_kernel(cwd=str(V3_ROOT), env=env)
    kc = km.client()
    kc.start_channels()
    try:
        kc.wait_for_ready(timeout=180)
        print(f"[nb] kernel ready (pid {km.provisioner.process.pid if km.provisioner else '?'})", flush=True)
        count = 0
        t_all = time.perf_counter()
        for idx, cell in enumerate(nb["cells"]):
            if cell["cell_type"] != "code":
                continue
            count += 1
            if max_cells is not None and count > max_cells:
                break
            code = _text(cell["source"])
            if not code.strip():
                cell["outputs"], cell["execution_count"] = [], None
                continue
            print(f"\n[nb] ---- cell {count} ({len(code)} chars) ----", flush=True)
            t0 = time.perf_counter()
            outputs, exec_count, err = _run_cell(kc, code, cell_timeout, km=km)
            cell["outputs"], cell["execution_count"] = outputs, exec_count
            dt = time.perf_counter() - t0
            print(f"[nb] ---- cell {count} done in {dt:.1f}s"
                  + (" (ERROR)" if err else "") + " ----", flush=True)
            if err:
                for o in outputs:
                    if o["output_type"] == "error":
                        _console("\n".join(o["traceback"]) + "\n")
                print(f"[nb] FAILED at cell {count}; notebook left partially executed", flush=True)
                break
        print(f"\n[nb] total {time.perf_counter() - t_all:.1f}s for {count} code cells", flush=True)
        out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
        print(f"[nb] wrote {out}", flush=True)
        return 0
    finally:
        kc.stop_channels()
        km.shutdown_kernel(now=True)


def _run_cell(kc, code: str, timeout: float, km=None):
    msg_id = kc.execute(code, store_history=True, allow_stdin=False)
    outputs: list[dict] = []
    exec_count = None
    err = False
    deadline = time.perf_counter() + timeout
    last_live = time.perf_counter()
    last_beat = time.perf_counter()

    def push(o: dict) -> None:
        if (o["output_type"] == "stream" and outputs
                and outputs[-1]["output_type"] == "stream"
                and outputs[-1]["name"] == o["name"]):
            outputs[-1]["text"] = _text(outputs[-1]["text"]) + _text(o["text"])
        else:
            outputs.append(o)

    idle = False
    while not idle:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            raise TimeoutError(f"cell exceeded {timeout:.0f}s")
        try:
            msg = kc.get_iopub_msg(timeout=30.0)
        except Empty:
            # the kernel is busy but silent (stdout/stderr can stay buffered for a
            # long time, e.g. tqdm bars written with '\r'); just keep waiting.
            if km is not None and not km.is_alive():
                raise RuntimeError("kernel died while executing a cell") from None
            if time.perf_counter() - last_beat > 300:
                print(f"[nb] ... still running ({time.perf_counter() - last_beat:.0f}s quiet)",
                      flush=True)
                last_beat = time.perf_counter()
            continue
        if msg["parent_header"].get("msg_id") != msg_id:
            continue
        mtype, content = msg["msg_type"], msg["content"]
        if mtype == "status":
            if content.get("execution_state") == "idle":
                idle = True
        elif mtype == "stream":
            push({"output_type": "stream", "name": content["name"], "text": content["text"]})
            now = time.perf_counter()
            text = _text(content["text"])
            if content["name"] == "stderr":
                if "\n" in text or now - last_live > 20.0:
                    _console(text)
                    last_live = now
            else:
                _console(text)
                last_live = now
        elif mtype == "execute_result":
            push({"output_type": "execute_result", "data": content["data"],
                  "metadata": content.get("metadata", {}),
                  "execution_count": content.get("execution_count")})
        elif mtype == "display_data":
            push({"output_type": "display_data", "data": content["data"],
                  "metadata": content.get("metadata", {})})
        elif mtype == "error":
            err = True
            push({"output_type": "error", "ename": content["ename"],
                  "evalue": content["evalue"], "traceback": content["traceback"]})
    while True:  # drain the shell reply for the execution count
        try:
            reply = kc.get_shell_msg(timeout=2.0)
        except Exception:  # noqa: BLE001
            break
        if reply["parent_header"].get("msg_id") == msg_id:
            exec_count = reply["content"].get("execution_count")
            break
    return outputs, exec_count, err


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--notebook", default=str(NOTEBOOK))
    ap.add_argument("--out", default=None)
    ap.add_argument("--cells", type=int, default=None, help="execute only the first N code cells")
    ap.add_argument("--timeout", type=float, default=6 * 3600.0, help="seconds per cell")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    if args.dry_run:
        print(f"would execute {args.notebook} (V3_QUICK={os.environ.get('V3_QUICK')})")
        return 0
    nb = Path(args.notebook)
    out = Path(args.out) if args.out else nb
    return execute(nb, args.cells, args.timeout, out)


if __name__ == "__main__":
    raise SystemExit(main())
