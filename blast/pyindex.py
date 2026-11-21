"""Static index of a Python file: its "units".

A unit is a function (kind F), a class header (H) or any other statement that
sits directly in a module or class body (S). The selector reasons about a
patch one unit at a time.
"""

from __future__ import annotations

import ast

FUNC = (ast.FunctionDef, ast.AsyncFunctionDef)


def _start(node) -> int:
    decorators = getattr(node, "decorator_list", None)
    return min([node.lineno] + [d.lineno for d in decorators]) if decorators else node.lineno


def bound_names(node) -> set[str]:
    """Names a module- or class-level statement binds."""
    names: set[str] = set()
    stack = [node]
    while stack:
        n = stack.pop()
        if isinstance(n, FUNC + (ast.ClassDef,)):
            names.add(n.name)
            if n is not node:
                continue
            if isinstance(n, FUNC):
                continue
        elif isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            names.add(n.id)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for alias in n.names:
                names.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(n, ast.Lambda):
            continue
        stack.extend(ast.iter_child_nodes(n))
    return names


def referenced(node) -> set[str]:
    names: set[str] = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name):
            names.add(n.id)
        elif isinstance(n, ast.Attribute):
            names.add(n.attr)
        elif isinstance(n, ast.alias):
            names.add(n.name.split(".")[-1])
            if n.asname:
                names.add(n.asname)
    return names


class Unit:
    __slots__ = ("kind", "scope", "name", "start", "end", "node", "names")

    def __init__(self, kind, scope, name, start, end, node, names):
        self.kind = kind  # F, H or S
        self.scope = scope  # qualname of the enclosing class, "" at module level
        self.name = name
        self.start = start
        self.end = end
        self.node = node
        self.names = names  # names the unit binds in its scope

    @property
    def key(self):
        return (self.kind, self.scope, self.name)


def units_of(tree: ast.Module) -> list[Unit]:
    units: list[Unit] = []

    def visit(body, scope):
        for node in body:
            if isinstance(node, FUNC):
                units.append(
                    Unit("F", scope, node.name, _start(node), node.end_lineno, node, {node.name})
                )
            elif isinstance(node, ast.ClassDef):
                qual = f"{scope}.{node.name}" if scope else node.name
                first = node.body[0].lineno if node.body else node.end_lineno + 1
                first = min(first, _start(node.body[0])) if node.body else first
                units.append(
                    Unit("H", scope, node.name, _start(node), max(first - 1, node.lineno), node, {node.name})
                )
                visit(node.body, qual)
            else:
                names = bound_names(node)
                label = ",".join(sorted(names)) or type(node).__name__
                units.append(Unit("S", scope, label, node.lineno, node.end_lineno, node, names))

    visit(tree.body, "")
    return units


def module_name(rel: str) -> str:
    name = rel[:-3].replace("/", ".")
    return name[: -len(".__init__")] if name.endswith(".__init__") else name


def _type_checking(test) -> bool:
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


def module_imports(tree: ast.Module, rel: str) -> set[str]:
    """Modules this file imports when it is loaded (dotted names, best effort).

    Imports inside functions and under `if TYPE_CHECKING:` do not run at load time.
    """
    package = module_name(rel).split(".")
    if not rel.endswith("__init__.py"):
        package = package[:-1]
    found: set[str] = set()
    stack = list(tree.body)
    while stack:
        n = stack.pop()
        if isinstance(n, FUNC) or isinstance(n, ast.Lambda):
            continue
        if isinstance(n, ast.If) and _type_checking(n.test):
            stack.extend(n.orelse)
            continue
        if isinstance(n, ast.Import):
            for alias in n.names:
                found.add(alias.name)
        elif isinstance(n, ast.ImportFrom):
            base = n.module or ""
            if n.level:
                prefix = package[: len(package) - (n.level - 1)]
                base = ".".join(prefix + ([base] if base else []))
            found.add(base)
            for alias in n.names:
                found.add(f"{base}.{alias.name}")
        else:
            stack.extend(ast.iter_child_nodes(n))
    return found


SIMPLE_TARGETS = (ast.Assign, ast.AugAssign, ast.AnnAssign)


def _attribute_targets(statement) -> tuple | None:
    """Attribute names a statement assigns to, or None when it also assigns to plain variables."""
    if not isinstance(statement, SIMPLE_TARGETS):
        return None
    targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
    names = []
    stack = list(targets)
    while stack:
        target = stack.pop()
        if isinstance(target, (ast.Tuple, ast.List)):
            stack.extend(target.elts)
        elif isinstance(target, ast.Starred):
            stack.append(target.value)
        elif isinstance(target, ast.Subscript):
            stack.append(target.value)
        elif isinstance(target, ast.Attribute):
            names.append(target.attr)
        else:
            return None
    return tuple(sorted(set(names))) or None


def _names_in(node):
    """(name, line) for every identifier an expression or statement mentions, strings included."""
    for n in ast.walk(node):
        if isinstance(n, ast.Name):
            yield n.id, n.lineno
        elif isinstance(n, ast.Attribute):
            # The load is attributed to the line the expression starts on; keep both ends.
            yield n.attr, n.lineno
            if n.end_lineno != n.lineno:
                yield n.attr, n.end_lineno
        elif isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value.isidentifier():
            yield n.value, n.lineno
        elif isinstance(n, ast.alias):
            yield n.name.split(".")[-1], n.lineno
            if n.asname:
                yield n.asname, n.lineno


# How a unit mentions a name:
DIRECT = 1  # in class- or module-level code, or in a signature: evaluated when the module loads
WHOLE = 2  # in a function's signature or decorators: matters to every call of the function


def references(unit: Unit):
    """Yield (name, line, mode). mode is DIRECT, WHOLE, or for a mention inside a function or
    lambda body: None (a plain use on that line) or a tuple of attribute names the line assigns to."""
    node = unit.node
    if unit.kind == "H":
        for part in node.bases + [k.value for k in node.keywords] + node.decorator_list:
            for name, line in _names_in(part):
                yield name, line, DIRECT
        return

    def body_refs(statements):
        for statement in statements:
            yield from statement_refs(statement)

    def statement_refs(statement):
        if isinstance(statement, FUNC):
            for part in statement.decorator_list + [statement.args] + ([statement.returns] if statement.returns else []):
                for name, line in _names_in(part):
                    yield name, line, None
            yield from body_refs(statement.body)
            return
        if isinstance(statement, ast.ClassDef):
            for name, line in _names_in(statement):
                yield name, line, None
            return
        targets = _attribute_targets(statement)
        blocks = []
        for field, value in ast.iter_fields(statement):
            if field in ("body", "orelse", "finalbody") and isinstance(value, list) and value and isinstance(value[0], ast.stmt):
                blocks.append(value)
            elif field == "handlers":
                for handler in value:
                    if handler.type is not None:
                        for name, line in _names_in(handler.type):
                            yield name, line, None
                    blocks.append(handler.body)
            elif field == "cases":
                for case in value:
                    for name, line in _names_in(case.pattern):
                        yield name, line, None
                    if case.guard is not None:
                        for name, line in _names_in(case.guard):
                            yield name, line, None
                    blocks.append(case.body)
            elif isinstance(value, ast.AST):
                for name, line in _names_in(value):
                    yield name, line, targets
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.AST):
                        for name, line in _names_in(item):
                            yield name, line, targets
        for block in blocks:
            yield from body_refs(block)

    if unit.kind == "F":
        for part in node.decorator_list + [node.args] + ([node.returns] if node.returns else []):
            for name, line in _names_in(part):
                yield name, line, WHOLE
        yield from body_refs(node.body)
        return

    # A class- or module-level statement: mentions inside lambda bodies run later, the rest run now.
    def split(n):
        if isinstance(n, ast.Lambda):
            for name, line in _names_in(n.args):
                yield name, line, DIRECT
            for name, line in _names_in(n.body):
                yield name, line, None
            return
        if isinstance(n, ast.Name):
            yield n.id, n.lineno, DIRECT
        elif isinstance(n, ast.Attribute):
            yield n.attr, n.lineno, DIRECT
        elif isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value.isidentifier():
            yield n.value, n.lineno, DIRECT
        elif isinstance(n, ast.alias):
            yield n.name.split(".")[-1], n.lineno, DIRECT
            if n.asname:
                yield n.asname, n.lineno, DIRECT
        for child in ast.iter_child_nodes(n):
            yield from split(child)

    yield from split(node)
