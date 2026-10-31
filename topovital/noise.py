"""NSTDB noise time blocks: Development / Test separation (brief §4)."""
from __future__ import annotations


class NoiseBlockError(ValueError):
    pass


def validate_blocks(dev, test, n_samples: int) -> None:
    if dev is None or test is None:
        raise NoiseBlockError("noise development/test blocks are pending (protocol values required)")
    for name, (s, e) in (("development", dev), ("test", test)):
        if not (0 <= s < e <= n_samples):
            raise NoiseBlockError(f"{name} block {(s, e)} outside [0, {n_samples}]")
    if dev[0] < test[1] and test[0] < dev[1]:
        raise NoiseBlockError(f"blocks overlap: dev={dev} test={test}")


def segment_in_block(start: int, length: int, block) -> bool:
    """Half-open [start, start+length) must lie entirely within the half-open block."""
    return block[0] <= start and start + length <= block[1]


def assert_segment(start: int, length: int, split: str, blocks: dict) -> None:
    if not segment_in_block(start, length, blocks[split]):
        raise NoiseBlockError(f"noise segment [{start},{start + length}) crosses/leaves the {split} block")
