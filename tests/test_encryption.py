"""Optional encryption at rest, and the cce encrypt / decrypt commands."""

import json
import sqlite3
from pathlib import Path

import pytest

pytest.importorskip("cryptography")

from cce_core.cli import main as cce_main  # noqa: E402
from cce_core.config import CCEConfig  # noqa: E402
from cce_core.crypto import Cipher, EncryptionError, PREFIX, convert_folder, open_cipher  # noqa: E402
from cce_core.engine import CCEEngine  # noqa: E402

from tests.test_live_sessions import PLANTED, make_config, make_messages  # noqa: E402

KEY = "correct horse battery staple"
QUESTION = "How many invoice retry attempts are allowed?"
SECRETS = ["seven attempts", "Marigold", "invoice retry limit"]


def cfg_for(tmp_path, key=KEY, **kw) -> CCEConfig:
    return make_config(tmp_path, retrieval_top_k=3, encryption_key=key, **kw)


def fill(cfg):
    mgr = CCEEngine(cfg).create_session_manager()
    for m in make_messages():
        mgr.add_message("s", m)
    mgr.pin("s", "The vault codeword is Marigold.")
    mgr.close("s")


def disk_bytes(base: Path) -> bytes:
    """Everything CCE wrote to the data folder."""
    return b"".join(p.read_bytes() for p in base.rglob("*") if p.is_file() and p.name != "sessions.json")


# ── The cipher ────────────────────────────────────────────────────────────────

def test_cipher_round_trip_and_tamper_detection(tmp_path):
    cipher = open_cipher(tmp_path, KEY, create=True)
    a, b = cipher.encrypt("hello"), cipher.encrypt("hello")
    assert a.startswith(PREFIX) and a != b                              # a fresh nonce each time
    assert cipher.decrypt(a) == "hello" and cipher.decrypt("plain text") == "plain text"
    assert cipher.encrypt(a) == a                                       # already encrypted: left alone
    broken = a[:-4] + ("AAAA" if not a.endswith("AAAA") else "BBBB")
    with pytest.raises(EncryptionError):
        cipher.decrypt(broken)
    other = Cipher(b"\x01" * 32)
    with pytest.raises(EncryptionError):
        other.decrypt(a)


def test_passphrase_rules(tmp_path):
    assert open_cipher(tmp_path / "plain", None, create=True) is None
    open_cipher(tmp_path / "enc", KEY, create=True)
    with pytest.raises(EncryptionError, match="CCE_ENCRYPTION_KEY"):
        open_cipher(tmp_path / "enc", None, create=True)
    with pytest.raises(EncryptionError, match="wrong passphrase"):
        open_cipher(tmp_path / "enc", "not the key", create=True)
    (tmp_path / "old").mkdir()
    with pytest.raises(EncryptionError, match="cce encrypt"):
        open_cipher(tmp_path / "old", KEY, create=False)


# ── A whole session under encryption ──────────────────────────────────────────

def test_nothing_readable_is_left_on_disk_and_everything_still_works(tmp_path):
    cfg = cfg_for(tmp_path)
    fill(cfg)
    blob = disk_bytes(cfg.base_dir)
    for secret in SECRETS:
        assert secret.encode() not in blob, secret
    assert b"billing" not in blob and b"hiking" not in blob                 # labels and search terms are sealed too
    assert (cfg.base_dir / "encryption.json").exists()

    mgr = CCEEngine(cfg).create_session_manager()                           # reopen with the key
    text = mgr.build_context_payload("s", QUESTION).to_string()
    assert PLANTED in text and "Marigold" in text
    dump = mgr.export_session("s")
    assert dump["pins"][0]["text"] == "The vault codeword is Marigold." and dump["notes"][0]["topic"]
    assert dump["session_summary"]["macro_text"]
    mgr.forget("s", start=0, end=3)
    mgr.edit_pin("s", dump["pins"][0]["pin_id"], "changed pin")
    mgr.close_all()
    after = disk_bytes(cfg.base_dir)
    assert b"changed pin" not in after


def test_live_data_is_sealed_while_a_session_is_open(tmp_path):
    cfg = cfg_for(tmp_path, hot_tier_max_turns=4)
    mgr = CCEEngine(cfg).create_session_manager()
    for m in make_messages():
        mgr.add_message("s", m)                                              # crash: no close()
    for secret in SECRETS[:1]:
        assert secret.encode() not in disk_bytes(cfg.base_dir)
    again = CCEEngine(cfg).create_session_manager()
    assert PLANTED in again.build_context_payload("s", QUESTION).to_string()


def test_wrong_or_missing_key_is_refused_with_a_clear_error(tmp_path):
    cfg = cfg_for(tmp_path)
    fill(cfg)
    with pytest.raises(EncryptionError, match="wrong passphrase"):
        CCEEngine(cfg_for(tmp_path, key="nope")).open_session("s")
    with pytest.raises(EncryptionError, match="encrypted"):
        CCEEngine(cfg_for(tmp_path, key=None)).open_session("s")


def test_key_comes_from_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("CCE_ENCRYPTION_KEY", KEY)
    assert CCEConfig().encryption_key == KEY
    monkeypatch.delenv("CCE_ENCRYPTION_KEY")
    assert CCEConfig().encryption_key is None


def test_a_key_does_not_silently_encrypt_an_existing_plain_folder(tmp_path):
    fill(cfg_for(tmp_path, key=None))
    with pytest.raises(EncryptionError, match="cce encrypt"):
        CCEEngine(cfg_for(tmp_path)).open_session("s")


# ── Converting a folder ───────────────────────────────────────────────────────

def test_encrypt_and_decrypt_a_folder_in_place(tmp_path, monkeypatch, capsys):
    plain = cfg_for(tmp_path, key=None)
    fill(plain)
    assert b"seven attempts" in disk_bytes(plain.base_dir)
    base = str(plain.base_dir)

    monkeypatch.setenv("CCE_ENCRYPTION_KEY", KEY)
    with pytest.raises(SystemExit) as e:
        cce_main(["encrypt", "--data-dir", base])
    assert e.value.code == 0 and "Encrypted" in capsys.readouterr().out
    blob = disk_bytes(plain.base_dir)
    assert all(s.encode() not in blob for s in SECRETS)
    mgr = CCEEngine(cfg_for(tmp_path)).create_session_manager()
    assert PLANTED in mgr.build_context_payload("s", QUESTION).to_string()
    mgr.close_all(checkpoint=False)

    with pytest.raises(EncryptionError, match="already encrypted"):
        convert_folder(plain.base_dir, KEY, encrypt=True)
    with pytest.raises(SystemExit):
        cce_main(["decrypt", "--data-dir", base])
    assert "Decrypted" in capsys.readouterr().out
    assert b"seven attempts" in disk_bytes(plain.base_dir) and not (plain.base_dir / "encryption.json").exists()
    mgr = CCEEngine(cfg_for(tmp_path, key=None)).create_session_manager()
    assert PLANTED in mgr.build_context_payload("s", QUESTION).to_string()


def test_convert_needs_the_key_and_a_folder(tmp_path):
    with pytest.raises(EncryptionError, match="CCE_ENCRYPTION_KEY"):
        convert_folder(tmp_path, None, encrypt=True)
    with pytest.raises(EncryptionError, match="no CCE data"):
        convert_folder(tmp_path, KEY, encrypt=True)
    fill(cfg_for(tmp_path, key=None))
    with pytest.raises(EncryptionError, match="not encrypted"):
        convert_folder(tmp_path / "data", KEY, encrypt=False)


def test_vectors_and_numbers_are_the_documented_exceptions(tmp_path):
    cfg = cfg_for(tmp_path)
    fill(cfg)
    con = sqlite3.connect(cfg.db_path)
    label, meso, meta, emb = con.execute("SELECT topic_label, meso_summary, metadata, embedding FROM memory_nodes").fetchone()
    con.close()
    assert label.startswith(PREFIX) and meso.startswith(PREFIX) and meta.startswith(PREFIX)
    assert isinstance(emb, bytes) and not emb.startswith(PREFIX.encode())            # the note vector is not encrypted


def test_forgotten_and_deleted_text_does_not_stay_in_the_file(tmp_path):
    """secure_delete: removed rows are overwritten, not left in free pages (checked on a plain folder)."""
    cfg = cfg_for(tmp_path, key=None)
    fill(cfg)
    mgr = CCEEngine(cfg).create_session_manager()
    mgr.forget("s", start=0, end=7)                              # the note with "seven attempts"
    mgr.close_all(checkpoint=False)
    assert b"seven attempts" not in disk_bytes(cfg.base_dir)
    CCEEngine(cfg).create_session_manager().delete("s")
    assert b"Marigold" not in cfg.db_path.read_bytes() and b"hiking" not in cfg.db_path.read_bytes()
