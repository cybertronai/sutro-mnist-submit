#!/usr/bin/env python3
"""Submit a kernel to a deployed submission site and wait for the verdict.

    uv run --python 3.12 python scripts/smoke.py <link> [--kernel submissions/ncm_baseline.py]
                            [--band mnist-medium-12pct] [--mode test] [--name smoke]
                            [--timeout 2700] [--expect pass|fail]

<link> is the secret URL printed by `uvx modal run web/app.py::link`. The
script posts the form the way a browser would, follows the redirect to the
new submission, polls its JSON endpoint every 15 s and prints the summary.
Exit code 0 when the run ends in the expected state (default: pass), 1
otherwise. Standard library only; any Python 3.7+ works, but the runbook
pins 3.12 like every other command.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def post_form(url: str, fields: dict) -> tuple[int, dict, bytes]:
    boundary = f"----smoke{uuid.uuid4().hex}"
    body = b""
    for name, value in fields.items():
        body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n").encode()
        body += value.encode("utf-8") + b"\r\n"
    body += f"--{boundary}--\r\n".encode()
    request = urllib.request.Request(url, data=body, method="POST", headers={
        "Content-Type": f"multipart/form-data; boundary={boundary}",
        "Content-Length": str(len(body)),
    })
    opener = urllib.request.build_opener(NoRedirect)
    try:
        with opener.open(request, timeout=120) as response:
            return response.status, dict(response.headers), response.read()
    except urllib.error.HTTPError as error:
        return error.code, dict(error.headers), error.read()


def get_json(url: str):
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            return json.loads(response.read())
    except (urllib.error.URLError, ValueError, TimeoutError) as error:
        return {"_transient": repr(error)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("link", help="the secret site URL, https://...modal.run/<token>")
    parser.add_argument("--kernel", default="submissions/ncm_baseline.py")
    parser.add_argument("--band", default="mnist-medium-12pct")
    parser.add_argument("--mode", default="test", choices=["test", "benchmark", "leaderboard"])
    parser.add_argument("--name", default="smoke test")
    parser.add_argument("--timeout", type=int, default=2700,
                        help="seconds to wait for a verdict (a leaderboard run may take up to 2280 s)")
    parser.add_argument("--expect", default="pass", choices=["pass", "fail"],
                        help="which verdict counts as success (ncm_baseline fails every band above test mode)")
    args = parser.parse_args()

    link = args.link.rstrip("/")
    source = Path(args.kernel).read_text()
    status, headers, body = post_form(f"{link}/submit", {
        "name": args.name, "band": args.band, "mode": args.mode, "source": source,
    })
    if status != 303:
        print(f"submit returned HTTP {status}; the page said:", file=sys.stderr)
        text = body.decode("utf-8", "replace")
        marker = text.find("Not submitted")
        print(text[marker:marker + 400] if marker >= 0 else text[:800], file=sys.stderr)
        return 1
    location = headers.get("Location") or headers.get("location")
    sid = location.rsplit("/", 1)[-1]
    print(f"submitted {sid}: {location}")

    deadline = time.time() + args.timeout
    record = None
    while time.time() < deadline:
        record = get_json(f"{link}/api/s/{sid}")
        if "_transient" in record:
            print(f"  poll error (will retry): {record['_transient']}")
        else:
            print(f"  {time.strftime('%H:%M:%S')} status={record.get('status')}")
            if record.get("status") != "queued":
                break
        time.sleep(15)
    if not record or record.get("status") in (None, "queued"):
        print("timed out waiting for a verdict", file=sys.stderr)
        return 1

    summary = record.get("summary") or {}
    print(json.dumps({
        "status": record.get("status"),
        "verdict": summary.get("verdict"),
        "mean_ms": summary.get("mean_ms"),
        "accuracy_pct": summary.get("accuracy_pct"),
        "correct": summary.get("correct"),
        "total": summary.get("total"),
        "required": summary.get("required"),
        "holdout_pct": summary.get("holdout_pct"),
        "gpu": record.get("gpu"),
        "harness": (summary.get("system") or {}).get("harness"),
        "charged_usd": record.get("charged_usd"),
        "reserved_usd": record.get("reserved_usd"),
        "billable_s": record.get("billable_s"),
        "error": record.get("error") or summary.get("error"),
    }, indent=2))
    if record.get("status") == "error":
        return 1
    wanted = "passed" if args.expect == "pass" else "failed"
    return 0 if record.get("status") == wanted else 1


if __name__ == "__main__":
    sys.exit(main())
