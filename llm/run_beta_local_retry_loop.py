#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path


DEFAULT_SEEDS = "326 380 366 386 327 318 383 308 334 310 309 332 335 356 323 307 325 368 361 320"
DEFAULT_COMPOSITIONS = "symmetric_n6 polarized_n6 three_clusters_n6"


def timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log_line(logf, message: str) -> None:
    line = f"[{timestamp()}] {message}"
    print(line, flush=True)
    print(line, file=logf, flush=True)


def newest_output_mtime(output_dir: Path) -> float | None:
    if not output_dir.exists():
        return None

    newest: float | None = None
    for path in output_dir.iterdir():
        if not path.is_file():
            continue
        mtime = path.stat().st_mtime
        if newest is None or mtime > newest:
            newest = mtime
    return newest


class ProgressTracker:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.lock = threading.Lock()
        self.last_progress_monotonic = time.monotonic()
        self.last_seen_output_mtime = newest_output_mtime(output_dir)

    def note_stream_output(self) -> None:
        with self.lock:
            self.last_progress_monotonic = time.monotonic()

    def refresh_output_progress(self) -> None:
        latest = newest_output_mtime(self.output_dir)
        if latest is None:
            return
        with self.lock:
            if self.last_seen_output_mtime is None or latest > self.last_seen_output_mtime:
                self.last_seen_output_mtime = latest
                self.last_progress_monotonic = time.monotonic()

    def seconds_since_progress(self) -> float:
        with self.lock:
            return time.monotonic() - self.last_progress_monotonic


class StreamPump(threading.Thread):
    def __init__(self, stream, logf, tracker: ProgressTracker):
        super().__init__(daemon=True)
        self.stream = stream
        self.logf = logf
        self.tracker = tracker

    def run(self) -> None:
        try:
            for line in iter(self.stream.readline, ""):
                if not line:
                    break
                self.tracker.note_stream_output()
                sys.stdout.write(line)
                sys.stdout.flush()
                self.logf.write(line)
                self.logf.flush()
        finally:
            self.stream.close()


class BetaRetrySupervisor:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.stop_signal: int | None = None
        self.current_process: subprocess.Popen[str] | None = None

    def request_stop(self, signum: int, _frame) -> None:
        self.stop_signal = signum
        process = self.current_process
        if process is not None and process.poll() is None:
            process.terminate()

    def build_env(self) -> dict[str, str]:
        env = os.environ.copy()
        env["LLM_BASE_URL"] = self.args.llm_base_url
        env["LLM_API_FLAVOR"] = self.args.llm_api_flavor
        env["LLM_MODEL"] = self.args.llm_model
        env["BETA_LOCAL_ROUNDS"] = str(self.args.rounds)
        env["PYTHONUNBUFFERED"] = "1"
        if self.args.beta_local_parallel:
            env["BETA_LOCAL_PARALLEL"] = self.args.beta_local_parallel
        if self.args.cloud_beta_parallel:
            env["CLOUD_BETA_PARALLEL"] = self.args.cloud_beta_parallel
        return env

    def build_make_command(self) -> list[str]:
        return [
            "make",
            "-f",
            "llm/Makefile",
            self.args.sweep_target,
            f"BETA_LOCAL_SEEDS={self.args.seeds}",
            f"BETA_LOCAL_COMPOSITIONS={self.args.compositions}",
            f"BETA_LOCAL_RUN_TAG_PREFIX={self.args.run_tag_prefix}",
            f"BETA_LOCAL_OUTPUT_DIR={self.args.output_dir}",
            f"BETA_LOCAL_RESUME={self.args.resume}",
        ]

    def terminate_child(self, process: subprocess.Popen[str], logf, reason: str) -> int:
        log_line(logf, reason)
        process.terminate()
        try:
            return process.wait(timeout=self.args.terminate_grace_s)
        except subprocess.TimeoutExpired:
            log_line(logf, "child did not exit after terminate; killing")
            process.kill()
            return process.wait()

    def run_streamed_process(
        self,
        command: list[str] | str,
        *,
        logf,
        env: dict[str, str] | None = None,
        shell: bool = False,
        stall_timeout_s: float | None,
    ) -> tuple[int, bool]:
        tracker = ProgressTracker(self.args.output_dir)
        process = subprocess.Popen(
            command,
            cwd=self.args.root_dir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            shell=shell,
            executable="/bin/bash" if shell else None,
        )
        self.current_process = process
        assert process.stdout is not None
        pump = StreamPump(process.stdout, logf, tracker)
        pump.start()

        stalled = False
        try:
            while True:
                if self.stop_signal is not None:
                    rc = self.terminate_child(process, logf, f"stop requested (signal={self.stop_signal})")
                    return rc, False

                rc = process.poll()
                if rc is not None:
                    return rc, stalled

                tracker.refresh_output_progress()
                if stall_timeout_s is not None and tracker.seconds_since_progress() > stall_timeout_s:
                    stalled = True
                    rc = self.terminate_child(
                        process,
                        logf,
                        f"beta sweep made no progress for {stall_timeout_s:.0f}s; terminating child for retry",
                    )
                    return rc, stalled

                time.sleep(self.args.poll_interval_s)
        finally:
            self.current_process = None
            pump.join(timeout=2)

    def maybe_run_success_hook(self, logf) -> int:
        if not self.args.on_success_cmd:
            return 0

        log_line(logf, f"running on-success command: {self.args.on_success_cmd}")
        rc, _ = self.run_streamed_process(
            self.args.on_success_cmd,
            logf=logf,
            env=os.environ.copy(),
            shell=True,
            stall_timeout_s=None,
        )
        if rc == 0:
            log_line(logf, "on-success command completed")
        else:
            log_line(logf, f"on-success command failed (rc={rc})")
        return rc

    def run(self) -> int:
        self.args.log_dir.mkdir(parents=True, exist_ok=True)
        self.args.output_dir.mkdir(parents=True, exist_ok=True)

        with self.args.log_file.open("a", buffering=1) as logf:
            command = self.build_make_command()
            env = self.build_env()

            if self.args.dry_run:
                log_line(
                    logf,
                    "dry run: "
                    + shlex.join(command)
                    + f" env[LLM_BASE_URL]={self.args.llm_base_url}"
                    + f" env[LLM_API_FLAVOR]={self.args.llm_api_flavor}"
                    + f" env[LLM_MODEL]={self.args.llm_model}",
                )
                return 0

            while True:
                log_line(
                    logf,
                    f"launching target={self.args.sweep_target} prefix={self.args.run_tag_prefix} "
                    f"rounds={self.args.rounds} seeds={self.args.seeds} comps={self.args.compositions}",
                )

                rc, stalled = self.run_streamed_process(
                    command,
                    logf=logf,
                    env=env,
                    shell=False,
                    stall_timeout_s=self.args.stall_timeout_s,
                )

                if self.stop_signal is not None or rc in (130, 143):
                    log_line(logf, f"beta sweep stopped intentionally (rc={rc})")
                    return rc

                if rc == 0:
                    log_line(logf, "beta sweep finished cleanly")
                    hook_rc = self.maybe_run_success_hook(logf)
                    return hook_rc

                if stalled:
                    log_line(logf, f"beta sweep exited with rc={rc} after stall; restarting with resume")
                else:
                    log_line(logf, f"beta sweep exited with rc={rc}; restarting with resume")

                if self.args.retry_delay_s > 0:
                    time.sleep(self.args.retry_delay_s)


def default_base_url(sweep_target: str) -> str:
    if sweep_target.startswith("townhall-beta-cloud-"):
        return "http://localhost:21435"
    return "http://localhost:20434"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-tag-prefix", default=os.getenv("RUN_TAG_PREFIX"))
    parser.add_argument("--seeds", default=os.getenv("SEEDS", DEFAULT_SEEDS))
    parser.add_argument("--compositions", default=os.getenv("COMPOSITIONS", DEFAULT_COMPOSITIONS))
    parser.add_argument("--rounds", type=int, default=int(os.getenv("BETA_LOCAL_ROUNDS", "8")))
    parser.add_argument("--beta-local-parallel", default=os.getenv("BETA_LOCAL_PARALLEL", ""))
    parser.add_argument("--cloud-beta-parallel", default=os.getenv("CLOUD_BETA_PARALLEL", ""))
    parser.add_argument("--sweep-target", default=os.getenv("SWEEP_TARGET", "townhall-beta-local-sweep"))
    parser.add_argument("--llm-base-url", default=os.getenv("LLM_BASE_URL"))
    parser.add_argument("--llm-api-flavor", default=os.getenv("LLM_API_FLAVOR", "openai"))
    parser.add_argument("--llm-model", default=os.getenv("LLM_MODEL", "gemma-4-E2B-it-Q8_0.gguf"))
    parser.add_argument("--on-success-cmd", default=os.getenv("ON_SUCCESS_CMD", ""))
    parser.add_argument("--log-dir", default=os.getenv("LOG_DIR", "outputs/llm_engine/logs"))
    parser.add_argument("--log-file", default=os.getenv("LOG_FILE"))
    parser.add_argument("--output-dir", default=os.getenv("OUTPUT_DIR"))
    parser.add_argument("--resume", default=os.getenv("BETA_LOCAL_RESUME", "true"))
    parser.add_argument(
        "--restore",
        dest="resume",
        action="store_const",
        const="true",
        help="Resume from existing checkpoints and outputs in the target directory.",
    )
    parser.add_argument(
        "--no-restore",
        dest="resume",
        action="store_const",
        const="false",
        help="Disable resume and start runs without the checkpoint resume flag.",
    )
    parser.add_argument("--retry-delay-s", type=float, default=float(os.getenv("RETRY_DELAY_S", "0")))
    parser.add_argument("--stall-timeout-s", type=float, default=float(os.getenv("STALL_TIMEOUT_S", "600")))
    parser.add_argument("--poll-interval-s", type=float, default=float(os.getenv("POLL_INTERVAL_S", "15")))
    parser.add_argument("--terminate-grace-s", type=float, default=float(os.getenv("TERMINATE_GRACE_S", "30")))
    parser.add_argument("--dry-run", action="store_true")
    return parser


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parent.parent
    args.root_dir = root_dir
    if not args.run_tag_prefix:
        args.run_tag_prefix = time.strftime("beta_local_%Y%m%d_%H%M%S")
    if not args.llm_base_url:
        args.llm_base_url = default_base_url(args.sweep_target)

    args.log_dir = (root_dir / args.log_dir).resolve()
    if args.output_dir:
        args.output_dir = (root_dir / args.output_dir).resolve()
    else:
        args.output_dir = (root_dir / "llm/traces/beta_local" / args.run_tag_prefix).resolve()

    if args.log_file:
        args.log_file = (root_dir / args.log_file).resolve()
    else:
        args.log_file = args.log_dir / f"{args.run_tag_prefix}_supervisor.log"

    args.on_success_cmd = args.on_success_cmd.strip()
    args.beta_local_parallel = args.beta_local_parallel.strip()
    args.cloud_beta_parallel = args.cloud_beta_parallel.strip()
    return args


def main() -> int:
    args = normalize_args(build_parser().parse_args())
    supervisor = BetaRetrySupervisor(args)
    signal.signal(signal.SIGINT, supervisor.request_stop)
    signal.signal(signal.SIGTERM, supervisor.request_stop)
    return supervisor.run()


if __name__ == "__main__":
    raise SystemExit(main())