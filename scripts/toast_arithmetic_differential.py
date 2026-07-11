"""Compare moo_interp arithmetic expressions with a live ToastStunt oracle.

Run from the repository root with:

    uv run --with ../moo-conformance-tests scripts/toast_arithmetic_differential.py
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from lambdamoo_db.database import ObjNum
from moo_conformance import SocketTransport

from moo_interp.errors import MOOError, MOOException
from moo_interp.list import MOOList
from moo_interp.map import MOOMap
from moo_interp.moo_ast import VMRunError, compile, parse, run
from moo_interp.string import MOOString


TOAST_ROOT = "/root/src/toaststunt"
TOAST_BINARY = f"{TOAST_ROOT}/build-release/moo"
TOAST_DATABASE = f"{TOAST_ROOT}/test/Test.db"


def normalize_value(value: Any) -> Any:
    if isinstance(value, ObjNum):
        return f"#{int(value)}"
    if isinstance(value, MOOError):
        return value.name
    if isinstance(value, (MOOString, str)):
        return str(value)
    if isinstance(value, MOOList):
        return [normalize_value(item) for item in value]
    if isinstance(value, MOOMap):
        return {
            normalize_value(key): normalize_value(value[key])
            for key in value
        }
    return value


def local_result(expression: str) -> dict[str, Any]:
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            vm = run(compile(parse(f"return {expression};")))
        return {"success": True, "value": normalize_value(vm.result)}
    except VMRunError as error:
        cause = error.message
        if isinstance(cause, MOOException):
            return {"success": False, "error": cause.error_code.name}
        return {"success": False, "error": type(cause).__name__}
    except MOOException as error:
        return {"success": False, "error": error.error_code.name}
    except ValueError as error:
        return {"success": False, "error": f"['Line 1:  {error}']"}


def toast_result(transport: SocketTransport, expression: str) -> dict[str, Any]:
    result = transport.execute(f"return {expression};")
    if result.success:
        return {"success": True, "value": result.value}
    if result.error is not None:
        return {"success": False, "error": result.error.value}
    return {"success": False, "error": result.error_message}


def arithmetic_expressions() -> list[str]:
    expressions: list[str] = []
    values = (-7, -2, -1, 0, 1, 2, 7)
    for left in values:
        for right in values:
            for operator in ("+", "-", "*", "/", "%"):
                expressions.append(f"{left} {operator} {right}")
            for operator in ("==", "!=", "<", "<=", ">", ">="):
                expressions.append(f"{left} {operator} {right}")
            for operator in ("|.", "&.", "^."):
                expressions.append(f"{left} {operator} {right}")

    for base in (-7, -2, -1, 0, 1, 2, 7):
        for exponent in (-4, -3, -1, 0, 1, 2, 5):
            expressions.append(f"{base} ^ {exponent}")

    for value in (-7, -1, 0, 1, 7):
        expressions.append(f"~{value}")
        for shift in (-1, 0, 1, 2, 63, 64, 65):
            expressions.append(f"{value} << {shift}")
            expressions.append(f"{value} >> {shift}")

    return expressions


def container_expressions() -> list[str]:
    return [
        '""',
        '"Toast"',
        '"abc" + "DEF"',
        '"AbC" == "aBc"',
        '"AbC" != "aBd"',
        '"b" in "abc"',
        '"z" in "abc"',
        '"toast"[1]',
        '"toast"[5]',
        '"toast"[2..4]',
        '"toast"[1..$]',
        '"toast"[^..$]',
        '""[1]',
        '{}',
        '{1, 2, 3}',
        '{1, 2} + {3, 4}',
        '2 in {1, 2, 3}',
        '4 in {1, 2, 3}',
        '{1, 2, 3}[1]',
        '{1, 2, 3}[3]',
        '{1, 2, 3}[2..3]',
        '{1, 2, 3}[1..$]',
        '{1, 2, 3}[^..$]',
        '{}[1]',
        '[]',
        '[1 -> "one", 2 -> "two"]',
        '[1 -> "one", 2 -> "two"][1]',
        '[1 -> "one", 2 -> "two"][2]',
        '[1 -> "one", 2 -> "two"][3]',
        '[3 -> "three", 1 -> "one", 2 -> "two"][^]',
        '[3 -> "three", 1 -> "one", 2 -> "two"][$]',
        '1 in [1 -> "one", 2 -> "two"]',
        '3 in [1 -> "one", 2 -> "two"]',
        '{1, {2, 3}, "x"}',
        '["list" -> {1, 2}, "map" -> [1 -> 2]]',
    ]


def conversion_expressions() -> list[str]:
    values = (
        "0",
        "-7",
        "1.5",
        '"42"',
        '"bad"',
        "#3",
        "E_TYPE",
        "{}",
        "[]",
        "true",
    )
    expressions = [f"typeof({value})" for value in values]
    expressions += [f"typename({value})" for value in values]
    expressions += [
        "is_type(1, INT)",
        "is_type(1.0, FLOAT)",
        'is_type("x", STR)',
        "is_type({}, LIST)",
        "is_type([], MAP)",
        "is_type(#3, OBJ)",
        "is_type(E_TYPE, ERR)",
        "is_type(1, FLOAT)",
        "toint(1)",
        "toint(-7.9)",
        'toint("42")',
        'toint("-7")',
        'toint("bad")',
        "toint(#3)",
        "toint(E_TYPE)",
        "tofloat(1.5)",
        "tofloat(1)",
        'tofloat("1.5")',
        'tofloat("bad")',
        "tofloat(true)",
        "tonum(7)",
        "tonum(1.5)",
        'tonum("42")',
        'tonum("1.5")',
        'tonum("bad")',
        "toobj(3)",
        "toobj(#3)",
        'toobj("#3")',
        'toobj("3")',
        'toobj("bad")',
        "toerr(1)",
        "toerr(E_TYPE)",
        'toerr("E_TYPE")',
        'toerr("TYPE")',
        'toerr("bad")',
        "tostr()",
        "tostr(1)",
        'tostr("a", 2, #3)',
        "toliteral(1)",
        "toliteral(1.5)",
        'toliteral("toast")',
        "toliteral(#3)",
        "toliteral(E_TYPE)",
        "toliteral({1, \"x\", #3})",
        'toliteral(["a" -> 1])',
    ]
    return expressions


def available_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait_for_server(port: int, process: subprocess.Popen[bytes]) -> None:
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"ToastStunt exited with status {process.returncode}")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.05)
    raise RuntimeError("ToastStunt did not open its port within 15 seconds")


def main() -> int:
    port = available_port()
    oracle_dir = f"/tmp/moo_interp_toast_{os.getpid()}"
    command = (
        f"set -eu; rm -rf {oracle_dir}; mkdir -p {oracle_dir}; "
        f"cp {TOAST_DATABASE} {oracle_dir}/oracle.db; "
        f"cd {oracle_dir}; exec {TOAST_BINARY} oracle.db oracle.out -p {port}"
    )
    process = subprocess.Popen(
        ["wsl.exe", "sh", "-lc", command],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    transport = SocketTransport(
        "127.0.0.1",
        port,
        ensure_standard_properties=False,
    )
    mismatches: list[dict[str, Any]] = []
    try:
        wait_for_server(port, process)
        transport.connect("wizard")
        expressions = (
            arithmetic_expressions()
            + container_expressions()
            + conversion_expressions()
        )
        for expression in expressions:
            local = local_result(expression)
            toast = toast_result(transport, expression)
            if local != toast:
                mismatches.append(
                    {"expression": expression, "local": local, "toast": toast}
                )

        summary = {
            "oracle": f"{TOAST_ROOT}@aecc51e",
            "cases": len(expressions),
            "matched": len(expressions) - len(mismatches),
            "mismatches": mismatches,
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 1 if mismatches else 0
    finally:
        transport.disconnect()
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        subprocess.run(
            ["wsl.exe", "sh", "-lc", f"rm -rf {oracle_dir}"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


if __name__ == "__main__":
    sys.exit(main())
