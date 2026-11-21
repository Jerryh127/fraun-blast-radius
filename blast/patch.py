"""Reads a unified diff (git format) and applies it to text, in pure Python."""

from __future__ import annotations

import re

HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


class FilePatch:
    def __init__(self):
        self.old: str | None = None  # path before, None when the file is new
        self.new: str | None = None  # path after, None when the file is deleted
        self.hunks: list[tuple[int, list[str]]] = []  # (old start line, body lines)
        self.binary = False
        self.mode_only = True


def _path(token: str) -> str | None:
    token = token.split("\t")[0].strip()
    if token == "/dev/null":
        return None
    if token.startswith('"') and token.endswith('"'):
        token = token[1:-1].encode().decode("unicode_escape").encode("latin-1").decode("utf-8", "replace")
    return token[2:] if token[:2] in ("a/", "b/") else token


def parse(text: str) -> list[FilePatch]:
    files: list[FilePatch] = []
    current: FilePatch | None = None
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("diff --git "):
            current = FilePatch()
            files.append(current)
            match = re.match(r"diff --git a/(.*) b/(.*)$", line)
            if match:
                current.old, current.new = match.group(1), match.group(2)
            i += 1
            continue
        if current is None:
            if line.startswith("--- ") and i + 1 < len(lines) and lines[i + 1].startswith("+++ "):
                current = FilePatch()
                files.append(current)
            else:
                i += 1
                continue
        if line.startswith("rename from "):
            current.old, current.mode_only = line[12:].strip(), False
        elif line.startswith("rename to "):
            current.new, current.mode_only = line[10:].strip(), False
        elif line.startswith("new file mode"):
            current.old, current.mode_only = None, False
        elif line.startswith("deleted file mode"):
            current.new, current.mode_only = None, False
        elif line.startswith(("Binary files ", "GIT binary patch")):
            current.binary, current.mode_only = True, False
        elif line.startswith("--- "):
            old = _path(line[4:])
            if old is None:
                current.old = None
            elif current.old is None and not any(lines[j].startswith("new file") for j in range(max(0, i - 4), i)):
                current.old = old
            current.mode_only = False
        elif line.startswith("+++ "):
            new = _path(line[4:])
            if new is None:
                current.new = None
            elif current.new is None and not any(lines[j].startswith("deleted file") for j in range(max(0, i - 5), i)):
                current.new = new
        else:
            match = HUNK.match(line)
            if match:
                old_start = int(match.group(1))
                old_left = int(match.group(2) or 1)
                new_left = int(match.group(4) or 1)
                body: list[str] = []
                i += 1
                while i < len(lines) and (old_left > 0 or new_left > 0):
                    row = lines[i]
                    tag = row[:1]
                    if tag == "\\":
                        body.append(row)
                    elif tag == "-":
                        old_left -= 1
                        body.append(row)
                    elif tag == "+":
                        new_left -= 1
                        body.append(row)
                    else:
                        old_left -= 1
                        new_left -= 1
                        body.append(" " + row[1:])
                    i += 1
                if i < len(lines) and lines[i].startswith("\\"):
                    body.append(lines[i])
                    i += 1
                current.hunks.append((old_start, body))
                current.mode_only = False
                continue
        i += 1
    return files


class PatchError(Exception):
    pass


def apply(old_text: str, hunks: list[tuple[int, list[str]]]) -> str:
    """Returns the patched text. Raises PatchError when the context does not match."""
    old_lines = old_text.split("\n")
    ends_with_newline = old_text.endswith("\n")
    if ends_with_newline:
        old_lines.pop()
    out: list[str] = []
    position = 0
    new_ends_with_newline = ends_with_newline
    for old_start, body in hunks:
        start = max(old_start - 1, 0) if old_lines or old_start else 0
        if start < position or start > len(old_lines):
            raise PatchError("hunk out of order")
        out.extend(old_lines[position:start])
        position = start
        last_tag = " "
        for row in body:
            tag, content = row[:1], row[1:]
            if tag == "\\":
                if last_tag in "+ ":
                    new_ends_with_newline = False
                if last_tag == "-":
                    pass
                continue
            if tag in " -":
                if position >= len(old_lines) or old_lines[position] != content:
                    raise PatchError("context mismatch")
                position += 1
                if tag == " ":
                    out.append(content)
            else:
                out.append(content)
                new_ends_with_newline = True
            last_tag = tag
            if tag in "+ " and position >= len(old_lines) and tag == " ":
                new_ends_with_newline = ends_with_newline
    out.extend(old_lines[position:])
    if position < len(old_lines):
        new_ends_with_newline = ends_with_newline
    text = "\n".join(out)
    return text + "\n" if (new_ends_with_newline and out) else text
