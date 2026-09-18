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
Challenge interface for Breaking RSA (semiprime factorization).
"""

from __future__ import annotations
from dataclasses import dataclass
import logging
from typing import *

import gmpy2

from . import Challenge, Serde

_logger = logging.getLogger(__name__)


@dataclass
class Problem(Serde):
    """
    Dataclass to hold problem instances of the `BreakingRSA` challenge.

    Attributes:
        difficulty (`int`):
            The difficulty level of the generated problem. This number is
            guaranteed to be non-negative.
        num (`int`):
            The semiprime to factor. This number is guaranteed to be at least 6.
        num_bits (`int`):
            The nominal bit-width of `num`. `num` is allowed to be less than `2
            ** (num_bits - 1)` (that is, `num` may have leading zeros in its
            binary expansion), but will always be less than `2 ** num_bits`.
    """
    difficulty: int
    num: int
    num_bits: int


@dataclass
class Solution(Serde):
    """
    Dataclass to hold the solution to a `BreakingRSA` problem.

    Attributes:
        status (`str`):
            Final solution status.
        p (`Optional[int]`):
            First prime factor.
        q (`Optional[int]`):
            Second prime factor.
    """
    status: str
    p: Optional[int]
    q: Optional[int]


@dataclass
class Verif(Serde):
    """
    Dataclass to hold additional data used for verification of a solution to a
    `BreakingRSA` problem.

    Attributes:
        n (`int`):
            Semiprime number.
        p (`int`):
            First prime factor.
        q (`int`):
            Second prime factor.
    """
    n: int
    p: int
    q: int


def validate_breaking_rsa_solution(
    solution: Solution, verif: Verif, prob: Problem | None = None, require_success_status: bool = True
) -> tuple[bool, str | None]:
    """Single source of truth for Breaking RSA solution correctness (factor match).

    This logic is shared to prevent divergence between:

    - BreakingRSA.verify (used by workbench for offline testing)
    - breaking_rsa_solution.run (used by the live validator for platform submissions)

    Returns (success, failure_reason or None). The reason is a human-readable
    string suitable for logging or returning to the platform.
    """
    if require_success_status and getattr(solution, "status", None) != "success":
        return False, f"Solution status is '{getattr(solution, 'status', None)}', not 'success'"

    if solution.p is None or solution.q is None:
        return False, "Solution has missing factors (p or q is null)"

    try:
        sol_p = int(solution.p)
        sol_q = int(solution.q)
    except (TypeError, ValueError):
        return False, f"Solution p/q are not valid integers: p={solution.p}, q={solution.q}"

    sol_p, sol_q = min(sol_p, sol_q), max(sol_p, sol_q)
    p_check, q_check = min(verif.p, verif.q), max(verif.p, verif.q)
    n = verif.n

    if sol_p * sol_q != n:
        return False, f"p * q != n: {sol_p} * {sol_q} != {n}"

    if sol_p != p_check or sol_q != q_check:
        return False, (
            f"Factors don't match expected: got ({sol_p}, {sol_q}), "
            f"expected ({p_check}, {q_check})"
        )

    if prob is not None:
        if prob.num != n or sol_p * sol_q != prob.num:
            return False, f"n mismatch between problem ({prob.num}) and verif ({n})"

    return True, None


def _gen_prime(num_bits: int, rng: gmpy2.random_state) -> gmpy2.mpz:
    """
    Generate a random prime number with bit width *at least* `num_bits`. This
    works by first generating a number `n` of at least `num_bits` bits, and then
    finding the closest prime number that is at least `n`.

    Args:
        num_bits: int
            Desired minimum bit width.
        rng: gmpy2.random_state
            Random generator.

    Returns:
        - The generated prime number.
    """
    n = gmpy2.mpz_urandomb(rng, num_bits)
    # make sure we have 1's in the most- and least- significant bits
    n |= (1 << (num_bits - 1)) | 1
    return gmpy2.next_prime(n)


@dataclass
class BreakingRSA(Challenge[Problem, Solution, Verif]):
    """
    The Breaking RSA challenge (semiprime factorization).

    Attributes:
        difficulty (`int`):
            The difficulty level of the generated problem. This number should be
            non-negative.
        num_bits(`int`):
            The nominal bit-width of the semiprime `num`. `num` is allowed to be
            less than `2 ** (num_bits - 1)` (that is, `num` may have leading
            zeros in its binary expansion), but will always be less than `2 **
            num_bits`.
    """
    difficulty: int
    num_bits: int

    def generate(self, seed: int) -> tuple[Problem, Verif]:
        rng = gmpy2.random_state(seed)
        bits_p = self.num_bits // 2
        bits_q = self.num_bits - bits_p
        fermat_thresh = 2 ** max(bits_p - 100, 1)
        while True:
            p = _gen_prime(bits_p, rng)
            q = _gen_prime(bits_q, rng)
            n = p * q
            if n.bit_length() == self.num_bits and abs(p - q) > fermat_thresh:
                break
        n = int(n)
        p = int(p)
        q = int(q)
        _logger.info(f"Generated {self.num_bits}-bit semiprime ({len(str(n))} digits)")
        problem = Problem(self.difficulty, n, self.num_bits)
        verif = Verif(n, p, q)
        return (problem, verif)

    def verify(self, prob: Problem, sol: Solution, verif: Verif) -> bool:
        success, reason = validate_breaking_rsa_solution(
            sol, verif, prob, require_success_status=False
        )
        if success:
            _logger.info("Verification SUCCESS")
        else:
            if sol.p is None or sol.q is None:
                _logger.info(
                    f"Verification FAILURE: missing factors, got p={sol.p} and"
                    + f" q={sol.q}, with solution status {sol.status}"
                )
            else:
                _logger.info(f"Verification FAILURE: {reason or 'factors do not match'}")
        return success
