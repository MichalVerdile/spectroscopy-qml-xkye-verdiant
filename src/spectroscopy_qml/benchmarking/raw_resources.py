"""Framework-independent, explicitly sampled process GPU memory telemetry."""

import functools
import os
import shutil
import subprocess
import threading
import time


def measure_gpu_memory(function):
    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        stop = threading.Event()
        samples = []
        errors = []
        enabled = str(kwargs.get("device", "cpu")).startswith("cuda")

        def sample():
            try:
                output = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-compute-apps=pid,used_gpu_memory",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    timeout=3,
                    stderr=subprocess.DEVNULL,
                )
                total = 0
                for line in output.strip().splitlines():
                    pid, memory = line.split(",")
                    if int(pid.strip()) == os.getpid():
                        total += int(memory.strip()) * 1024 * 1024
                samples.append(total)
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                errors.append(type(error).__name__)

        def poll():
            while not stop.wait(1.0):
                sample()

        thread = None
        if enabled and shutil.which("nvidia-smi"):
            sample()
            thread = threading.Thread(target=poll, daemon=True)
            thread.start()
        started, cpu_started = time.perf_counter(), time.process_time()
        try:
            result = function(*args, **kwargs)
        finally:
            stop.set()
            if thread is not None:
                thread.join(timeout=4)
                sample()
        statistics = result[-1] if isinstance(result, tuple) else result
        statistics.update(
            fit_wall_seconds_this_invocation=time.perf_counter() - started,
            cpu_process_seconds_this_invocation=time.process_time() - cpu_started,
            gpu_process_peak_sampled_bytes=max(samples) if samples else None,
            gpu_process_memory_samples=len(samples),
            gpu_process_memory_errors=sorted(set(errors)),
            gpu_process_memory_scope="nvidia-smi process memory across devices, 1s sampling; may miss brief peaks; null if unavailable/CPU",
        )
        return result

    return wrapped
