"""Replay a serialized KV link feeding one unbatched decode slot (chapter 25)."""

import argparse
from dataclasses import dataclass
from math import isfinite


@dataclass
class Event:
    ready: float
    send_start: float
    send_end: float
    decode_start: float
    second_token: float


def replay(ready_times, transfer_ms, decode_ms):
    events = []
    link_free = decode_free = 0.0
    for ready in ready_times:
        send_start = max(ready, link_free)
        link_free = send_start + transfer_ms
        decode_start = max(link_free, decode_free)
        decode_free = decode_start + decode_ms
        events.append(Event(ready, send_start, link_free, decode_start, decode_free))
    return events


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ready", type=float, nargs="+", default=[0, 0, 4, 4],
                        help="first-token emission times in milliseconds, in FIFO order")
    parser.add_argument("--transfer-ms", type=float, default=4)
    parser.add_argument("--decode-ms", type=float, default=3)
    parser.add_argument("--deadline-ms", type=float, default=10)
    args = parser.parse_args()
    numbers = args.ready + [args.transfer_ms, args.decode_ms, args.deadline_ms]
    if not all(isfinite(value) and value >= 0 for value in numbers):
        parser.error("times must be finite and nonnegative")
    if sorted(args.ready) != args.ready:
        parser.error("--ready must be in chronological order")
    events = replay(args.ready, args.transfer_ms, args.decode_ms)
    print("request  ready   link-start  link-end  decode-start  second  gap (ms)")
    passed = 0
    for number, event in enumerate(events, 1):
        gap = event.second_token - event.ready
        passed += gap <= args.deadline_ms
        print(f"{number:7d} {event.ready:6g} {event.send_start:11g} {event.send_end:9g} "
              f"{event.decode_start:13g} {event.second_token:7g} {gap:8g}")
    print(f"First-to-second-token deadline met by {passed}/{len(events)}")


if __name__ == "__main__":
    main()
