"""Offline trust-receipt verifier (standalone CLI).

Usage:
    python scripts/verify_receipt.py <receipt.json> <public_key_b64_file>

Prints ``CLEAN`` and exits 0 if the receipt's signature is valid under the given public key;
prints ``TAMPERED`` and exits 1 otherwise (bad signature, wrong key, or any altered field).
No network, no DB — this is the whole point: anyone can verify a receipt with just the two files."""

from __future__ import annotations

import sys
from pathlib import Path

from rag_common.trust_receipt import TrustReceipt, verify_receipt

_EXPECTED_ARGC = 3  # prog + receipt.json + public_key_b64_file


def main(argv: list[str]) -> int:
    if len(argv) != _EXPECTED_ARGC:
        print("usage: python scripts/verify_receipt.py <receipt.json> <public_key_b64_file>", file=sys.stderr)
        return 2
    receipt = TrustReceipt.model_validate_json(Path(argv[1]).read_text(encoding="utf-8"))
    public_key_b64 = Path(argv[2]).read_text(encoding="utf-8").strip()
    if verify_receipt(receipt, public_key_b64):
        print("CLEAN")
        return 0
    print("TAMPERED")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
