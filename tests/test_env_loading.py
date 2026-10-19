from eplanes.config import read_env_file


def test_env_parser_variants(tmp_path):
    p = tmp_path / ".env"
    p.write_bytes("﻿# comment\r\nPIPISTREL_FLEET_A_USER = alice\r\nexport PIPISTREL_FLEET_A_PASSWORD='p@ss=word'\r\nEMPTY=\r\n".encode())
    env = read_env_file(p)
    assert env["PIPISTREL_FLEET_A_USER"] == "alice"
    assert env["PIPISTREL_FLEET_A_PASSWORD"] == "p@ss=word"
    assert env["EMPTY"] == ""
