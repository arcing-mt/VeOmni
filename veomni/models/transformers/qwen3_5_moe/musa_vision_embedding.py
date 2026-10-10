"""Guard one fixed Qwen3.5-MoE vision interpolation call site."""

import ast
import copy
import functools
import hashlib
import inspect
import os
import textwrap

from .vision_embedding_csr import cached_embedding


METHOD_SOURCE_SHA256 = "70159a4de78d0b0beba6ae926f3f7cc4f08caf6b1b23b06436269bfa72f495da"
_installed = False


def make_candidate(original):
    source = textwrap.dedent(inspect.getsource(original))
    method = ast.parse(source).body[0]
    # Hash exact dedented source, rather than AST dump fields which differ
    # between the local Python and the container's Python 3.10.
    digest = hashlib.sha256(source.encode()).hexdigest()
    if digest != METHOD_SOURCE_SHA256:
        raise RuntimeError(f"Vision helper changed; refuse cached indices: {digest}")
    assert not method.decorator_list and not original.__closure__

    class Replace(ast.NodeTransformer):
        count = 0

        def visit_Call(self, node):
            if ast.unparse(node.func) == "self.pos_embed":
                assert len(node.args) == 1 and ast.unparse(node.args[0]) == "indices" and not node.keywords
                self.count += 1
                return ast.copy_location(
                    ast.Call(
                        func=ast.Name(id="_trial_cached_embedding", ctx=ast.Load()),
                        args=[
                            node.func,
                            node.args[0],
                            ast.Tuple(
                                elts=[ast.Name(id=n, ctx=ast.Load()) for n in ("h", "w", "num_grid_per_side")],
                                ctx=ast.Load(),
                            ),
                        ],
                        keywords=[],
                    ),
                    node,
                )
            return self.generic_visit(node)

    replacer = Replace()
    method = replacer.visit(copy.deepcopy(method))
    assert replacer.count == 1
    namespace = dict(original.__globals__, _trial_cached_embedding=cached_embedding)
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    exec(compile(module, inspect.getfile(original), "exec"), namespace)
    return functools.update_wrapper(namespace[method.name], original)


def install(modeling):
    global _installed
    if _installed:
        return
    cls = modeling.Qwen3_5MoeVisionModel
    original = cls.fast_pos_embed_interpolate
    candidate = make_candidate(original)

    @functools.wraps(original)
    def interpolate(self, *args, **kwargs):
        if os.environ.get("VEOMNI_MUSA_VISION_EMBEDDING_CSR", "0") == "1":
            return candidate(self, *args, **kwargs)
        return original(self, *args, **kwargs)

    cls.fast_pos_embed_interpolate = interpolate
    _installed = True
