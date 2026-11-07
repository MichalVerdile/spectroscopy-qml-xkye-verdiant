"""Runtime tripwire against accidentally invoking historical preprocessing."""

import sys
from contextlib import contextmanager


@contextmanager
def strict_raw_guard():
    previous = sys.getprofile()

    def guard(frame, event, arg):
        module = frame.f_globals.get("__name__", "")
        if event == "call" and module == "spectroscopy_qml.benchmarking.preprocessing":
            raise RuntimeError("Preprocessing function called inside strict raw_native execution")
        if previous is not None:
            previous(frame, event, arg)

    sys.setprofile(guard)
    try:
        yield
    finally:
        sys.setprofile(previous)
