# The MIT License (MIT)
# Copyright © 2026 qBitTensor Labs
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

"""
High-level Challenge and Solver base class definitions.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import collections.abc
import json
from pathlib import Path
import typing
from typing import *


P = TypeVar("P")
S = TypeVar("S")
V = TypeVar("V")


class Challenge(Generic[P, S, V], ABC):
    """
    Base class for any challenge type.

    The type parameters `P` and `S` are generic over data types representing
    problem and solution instances, while `T` is a container for "secret"
    information to be retained for verification.

    >>> from dataclasses import dataclass
    >>> import secrets
    >>>
    >>> @dataclass
    >>> class Problem:
    >>>     # ...
    >>>
    >>> @dataclass
    >>> class Solution:
    >>>     # ...
    >>>
    >>> @dataclass
    >>> class Verif:
    >>>     # ...
    >>>
    >>> class MyChallenge(Challenge[Problem, Solution, Verif]):
    >>>     # ...
    >>>
    >>> # suppose `MySolver` is a sublass of `Solver[Problem, Solution]`
    >>>
    >>> # initialize a challenge instance with relevant parameters
    >>> challenge = MyChallenge(...)
    >>> # initialize a solver with relevant parameters
    >>> solver = MySolver(...)
    >>>
    >>> # solve the challenge!
    >>> seed = secrets.randbits(256)
    >>> (problem, secrets) = challenge.generate(seed)
    >>> solution = solver.solve(problem)
    >>> # verify solution
    >>> successful_solve = challenge.verify(problem, solution, secrets)
    """

    @abstractmethod
    def generate(self, seed: int) -> tuple[P, V]:
        """
        Generate a problem instance.

        Args:
            seed (`int`):
                Initial seed for all relevant randomization.

        Returns:
            - Problem instance.
            - Secrets retained for verification.
        """
        raise NotImplementedError()

    @abstractmethod
    def verify(self, problem: P, solution: S, secrets: V) -> bool:
        """
        Verify a solution to a problem instance, returning `True` for a positive
        result.

        Args:
            problem (`P`):
                The problem instance that `solution` is purported to solve.
            solution (`S`):
                The solution to `problem`.
            secrets (`V`):
                Additional information from `self.generate` retained for
                verification.

        Returns:
            - `True` if `solution` is a valid solution to `problem`.
        """
        raise NotImplementedError()


class Solver(Generic[P, S], ABC):
    """
    Solution counterpart to `Challenge[P, S, _]`.

    The type parameters `P` and `S` are generic over data types representing
    problem and solution instances; they should be made concrete for a child
    class implementing a solver for a particular challenge.

    >>> import secrets
    >>>
    >>> class MySolver(Solver[Problem, Solution]):
    >>>     # ...
    >>>
    >>> # suppose `MyChallenge` is a sublass of `Challenge[Problem, Solution, Verif]`
    >>>
    >>> # initialize a challenge instance with relevant parameters
    >>> challenge = MyChallenge(...)
    >>> # initialize a solver with relevant parameters
    >>> solver = MySolver(...)
    >>>
    >>> # solve the challenge!
    >>> seed = secrets.randbits(256)
    >>> (problem, secrets) = challenge.generate(seed)
    >>> solution = solver.solve(problem)
    >>> # verify solution
    >>> successful_solve = challenge.verify(problem, solution, secrets)
    """

    @abstractmethod
    def solve(self, problem: P) -> S:
        """
        Solve a given problem instance.

        Args:
            problem (`P`):
                The problem instance to solve.

        Returns:
            - The solution to `problem`.
        """
        raise NotImplementedError()


def _is_callable_hint(ty: Any) -> bool:
    """Check if a type hint represents a Callable type."""
    return (
        typing.get_origin(ty) is collections.abc.Callable
        or ty is collections.abc.Callable
    )


def _is_serde(ty: type) -> bool:
    """Check if `ty` is a concrete subclass of `Serde`."""
    return isinstance(ty, type) and issubclass(ty, Serde) and ty is not Serde


def _convert_from(val: Any, ty: Any) -> Any:
    """
    Recursively convert `val` to match `ty`, constructing nested `Serde`
    subclasses from dicts as needed.
    """
    if ty is Any:
        return val

    origin = typing.get_origin(ty)
    args = typing.get_args(ty)

    if origin is Union:
        if val is None and type(None) in args:
            return None
        for variant in args:
            if variant is type(None):
                continue
            try:
                return _convert_from(val, variant)
            except (TypeError, KeyError):
                continue
        raise TypeError(
            f"value {val!r} does not match any variant of {ty}"
        )
    if origin is list:
        if not isinstance(val, list):
            raise TypeError(f"expected list, got {type(val)}")
        if args:
            return [_convert_from(item, args[0]) for item in val]
        return val
    if origin is dict:
        if not isinstance(val, dict):
            raise TypeError(f"expected dict, got {type(val)}")
        if len(args) == 2:
            return {
                _convert_from(k, args[0]): _convert_from(v, args[1])
                for k, v in val.items()
            }
        return val
    if origin is tuple:
        if not isinstance(val, (list, tuple)):
            raise TypeError(f"expected tuple, got {type(val)}")
        if args:
            return tuple(_convert_from(item, a) for item, a in zip(val, args))
        return tuple(val)
    if _is_serde(ty):
        if isinstance(val, ty):
            return val
        if isinstance(val, dict):
            return ty.from_dict(val)
        raise TypeError(f"expected dict or {ty.__name__}, got {type(val)}")
    if isinstance(ty, type) and not isinstance(val, ty):
        raise TypeError(
            f"expected type `{ty.__name__}` but got `{type(val).__name__}`"
        )
    return val


def _convert_to(val: Any) -> Any:
    """
    Recursively convert `val` for serialization, turning nested `Serde`
    instances into dicts.
    """
    if isinstance(val, Serde):
        return val.to_dict()
    if isinstance(val, list):
        return [_convert_to(item) for item in val]
    if isinstance(val, dict):
        return {k: _convert_to(v) for k, v in val.items()}
    if isinstance(val, tuple):
        return [_convert_to(item) for item in val]
    return val


class Serde:
    """
    Helper base class to handle conversion of (mainly) simple dataclasses to and
    from ordinary dictionaries. Base dictionary conversion is then extended to
    allow for JSON (de)serialization.

    All conversion methods determine an expected data set from annotated
    attributes via `typing.get_type_hints`, which resolves string annotations
    (including those deferred by `from __future__ import annotations`) into real
    type objects. This means any attributes *not* annotated with a type will not
    be processed.

    A dataclass-like `__init__` constructor is also assumed by `from_*` methods,
    which is expected to take arguments in the same order and of the same types
    as the annotated attributes.

    Nested `Serde` subclasses are automatically constructed from dicts during
    deserialization and converted back to dicts during serialization.

    >>> from dataclasses import dataclass
    >>>
    >>> @dataclass
    >>> class Inner(Serde):
    >>>     x: int
    >>>
    >>> @dataclass
    >>> class Outer(Serde):
    >>>     a: int | float
    >>>     b: str = "hello"
    >>>     inner: Inner
    >>>     c = None # excluded: no type annotation
    >>>
    >>> Outer.from_dict({"a": 3.14, "b": "goodbye", "inner": {"x": 1}})
    Outer(a=3.14, b="goodbye", inner=Inner(x=1))
    >>> Outer.from_dict({"b": "good evening"})
    KeyError: missing expected key 'a'
    >>> dict_data = {"a": 1, "b": "goodbye", "inner": {"x": 1}}
    >>> assert Outer.from_dict(dict_data).to_dict() == dict_data
    """

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """
        Construct from a basic dictionary. Values are type-checked and
        recursively converted: nested `Serde` subclasses are constructed from
        sub-dicts, and generic containers (`list`, `dict`, `tuple`) have their
        elements recursively processed as well.

        Args:
            data (`dict[str, Any]`):
                Base dictionary values.

        Returns:
            - Constructed data class.

        Raises:
            - `KeyError` if `data` lacks a key with no default value.
            - `TypeError` if a value under a given key does not match its
              expected type.
        """
        hints = typing.get_type_hints(cls)
        defaults = {
            key: val
            for key in dir(cls)
            if (
                not key.startswith("__")
                and not isinstance(val := getattr(cls, key), Callable)
            )
        }
        args = list()
        for key, ty in hints.items():
            if _is_callable_hint(ty):
                continue
            if key in data:
                val = data[key]
            elif key in defaults:
                val = defaults[key]
            else:
                raise KeyError(f"missing expected key '{key}'")
            args.append(_convert_from(val, ty))
        return cls(*args)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert `self` into an ordinary dictionary. Nested `Serde` instances
        are recursively converted to dicts.

        Returns:
            - Untyped dictionary values. Guaranteed to have keys for all
              annotated attributes, excluding any `Callable` items.
        """
        hints = typing.get_type_hints(type(self))
        result = {}
        for key, ty in hints.items():
            if _is_callable_hint(ty):
                continue
            result[key] = _convert_to(getattr(self, key))
        return result

    @classmethod
    def from_json(cls, json_str: str) -> Self:
        """
        Parse from a source JSON string with `json.loads` and then construct
        from the resulting dictionary with `self.from_dict`.

        Args:
            json_str (`str`):
                JSON string.

        Returns:
            - Constructed data class.

        Raises:
            - `KeyError` if the input lacks a key with no default value.
            - `TypeError` if a value under a given key does not match its
              expected type.
        """
        return cls.from_dict(json.loads(json_str))

    def to_json(self) -> str:
        """
        Convert `self` into a JSON string.

        Returns:
            - JSON encoding of `self`.
        """
        return json.dumps(self.to_dict())

    @classmethod
    def from_json_file(cls, json_file: Path) -> Self:
        """
        Parse from a source JSON file with `json.load` and then construct from
        the resulting dictionary with `self.from_dict`.

        Args:
            json_file (`Path`):
                Path to the JSON file.

        Returns:
            - Constructed data class.

        Raises:
            - `KeyError` if the file contents lack a key with no default value.
            - `TypeError` if a value under a given key does not match its
              expected type.
            - ...exceptions raisable by `Path.open` or `json.load`.
        """
        with json_file.open("r") as infile:
            return cls.from_dict(json.load(infile))

    def to_json_file(self, out: Path) -> None:
        """
        Write `self` as a JSON string to a file.

        Args:
            out: (`Path`):
                Path to output JSON file.

        Raises:
            - ...exceptions raisable by `Path.open` or `json.dump`.
        """
        with out.open("w") as outfile:
            json.dump(self.to_dict(), outfile)
