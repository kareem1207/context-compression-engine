"""
Optional encryption at rest (extra: `encrypt`, which installs `cryptography`).

Set a passphrase in `CCE_ENCRYPTION_KEY` (or `CCEConfig(encryption_key=...)`). A key is derived from it with
scrypt and a random salt kept in `<data dir>/encryption.json`; text is encrypted with AES-256-GCM.

Encrypted: message text, note labels and summaries, the per-note metadata (which holds the search terms, the
vectors of messages and sentences, and times), pins and the session summaries in the cold tier.
NOT encrypted: the note embedding (one vector per note), turn numbers and token counts, session ids and the
session registry. Vectors can reveal what a note is roughly about. While CCE runs, the decrypted notes are held in
memory by the search index. Losing the passphrase loses the data.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import threading
from pathlib import Path

PREFIX = "enc1:"
_FILE = "encryption.json"
_CHECK_TEXT = "cce-encryption-check"
_lock = threading.Lock()
_keys: dict[tuple[str, str], bytes] = {}


class EncryptionError(ValueError):
    """The data folder's encryption does not match the passphrase that was given."""


class Cipher:
    def __init__(self, key: bytes):
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        except ImportError as exc:                                               # pragma: no cover
            raise ImportError(
                "encryption needs the `cryptography` package: pip install 'context-compression-engine[encrypt]'"
            ) from exc
        self._aes = AESGCM(key)

    def encrypt(self, text: str) -> str:
        if text.startswith(PREFIX):
            return text                                                            # already encrypted
        nonce = os.urandom(12)
        return PREFIX + base64.b64encode(nonce + self._aes.encrypt(nonce, text.encode("utf-8"), b"cce")).decode("ascii")

    def decrypt(self, text: str) -> str:
        if not text.startswith(PREFIX):
            return text                                                            # plaintext (not yet migrated)
        raw = base64.b64decode(text[len(PREFIX):])
        try:
            return self._aes.decrypt(raw[:12], raw[12:], b"cce").decode("utf-8")
        except Exception as exc:
            raise EncryptionError("could not decrypt: wrong passphrase, or the data was changed") from exc


def _derive(passphrase: str, salt: bytes) -> bytes:
    cache_key = (passphrase, salt.hex())
    with _lock:
        if cache_key not in _keys:
            _keys[cache_key] = hashlib.scrypt(passphrase.encode("utf-8"), salt=salt, n=2**14, r=8, p=1, dklen=32)
        return _keys[cache_key]


def is_encrypted_folder(base_dir: Path) -> bool:
    return (Path(base_dir) / _FILE).exists()


def open_cipher(base_dir: Path, passphrase: str | None, *, create: bool) -> Cipher | None:
    """
    The cipher for a data folder.
      encrypted folder, right passphrase -> Cipher
      encrypted folder, none / wrong     -> EncryptionError
      plaintext folder, no passphrase    -> None
      plaintext folder, passphrase       -> a new key is set up if `create`, else EncryptionError
    """
    path = Path(base_dir) / _FILE
    if path.exists():
        if not passphrase:
            raise EncryptionError(f"{base_dir} is encrypted: set CCE_ENCRYPTION_KEY to its passphrase.")
        info = json.loads(path.read_text(encoding="utf-8"))
        cipher = Cipher(_derive(passphrase, bytes.fromhex(info["salt"])))
        try:
            ok = cipher.decrypt(info["check"]) == _CHECK_TEXT
        except EncryptionError:
            ok = False
        if not ok:
            raise EncryptionError("wrong passphrase for this data folder.")
        return cipher
    if not passphrase:
        return None
    if not create:
        raise EncryptionError(
            f"{base_dir} holds data that is not encrypted. Encrypt it first: cce encrypt --data-dir {base_dir}"
        )
    salt = os.urandom(16)
    cipher = Cipher(_derive(passphrase, salt))
    Path(base_dir).mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"format": "aes-256-gcm/scrypt-n16384", "salt": salt.hex(), "check": cipher.encrypt(_CHECK_TEXT)}),
                    encoding="utf-8")
    return cipher


def convert_folder(base_dir: Path, passphrase: str | None, *, encrypt: bool) -> dict:
    """
    Encrypt (or decrypt) every text column and cold-tier file of an existing data folder in place.
    Stop everything that uses the folder first, and keep a copy: a crash in the middle of the cold-tier
    files leaves a mix, which still opens (plaintext is read as it is) and can be converted again.
    """
    import sqlite3

    base_dir = Path(base_dir)
    db_path = base_dir / "warm.db"
    if not passphrase:
        raise EncryptionError("set CCE_ENCRYPTION_KEY to the passphrase first.")
    if not db_path.exists():
        raise EncryptionError(f"no CCE data in {base_dir} (looked for warm.db).")
    already = is_encrypted_folder(base_dir)
    if encrypt and already:
        raise EncryptionError("this folder is already encrypted.")
    if not encrypt and not already:
        raise EncryptionError("this folder is not encrypted.")
    cipher = open_cipher(base_dir, passphrase, create=True)
    apply = cipher.encrypt if encrypt else cipher.decrypt

    counts = {"notes": 0, "turns": 0, "pins": 0, "cold_files": 0}
    con = sqlite3.connect(db_path)
    try:
        con.execute("PRAGMA secure_delete = ON")                  # overwrite the old pages instead of leaving them behind
        con.execute("BEGIN")
        for node_id, label, meso, micro, meta in con.execute(
                "SELECT node_id, topic_label, meso_summary, micro_summaries, metadata FROM memory_nodes").fetchall():
            con.execute("UPDATE memory_nodes SET topic_label = ?, meso_summary = ?, micro_summaries = ?, metadata = ? WHERE node_id = ?",
                        (apply(label), apply(meso), apply(micro), apply(meta), node_id))
            counts["notes"] += 1
        for session_id, idx, content, meta in con.execute("SELECT session_id, idx, content, metadata FROM turns").fetchall():
            con.execute("UPDATE turns SET content = ?, metadata = ? WHERE session_id = ? AND idx = ?",
                        (apply(content), apply(meta), session_id, idx))
            counts["turns"] += 1
        for pin_id, text in con.execute("SELECT pin_id, text FROM pins").fetchall():
            con.execute("UPDATE pins SET text = ? WHERE pin_id = ?", (apply(text), pin_id))
            counts["pins"] += 1
        con.commit()
        con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        con.execute("VACUUM")                                      # rewrite the file: no old text in free pages
    except BaseException:
        con.rollback()
        raise
    finally:
        con.close()

    for path in sorted((base_dir / "cold").glob("*.json")) if (base_dir / "cold").exists() else []:
        path.write_text(apply(path.read_text(encoding="utf-8")), encoding="utf-8")
        counts["cold_files"] += 1
    if not encrypt:
        (base_dir / _FILE).unlink()
    return counts
