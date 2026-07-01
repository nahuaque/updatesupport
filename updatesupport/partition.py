"""Finite partitions used as update supports."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Hashable, Iterable, Mapping


class PartitionError(ValueError):
    """Raised when a partition is malformed."""


@dataclass(frozen=True, eq=False)
class Partition:
    """A partition of a finite state space.

    Blocks are stored as tuples for stable display, while equality and hashing
    are set-based so block order does not matter.
    """

    blocks: tuple[tuple[Hashable, ...], ...]

    @classmethod
    def from_blocks(
        cls,
        blocks: Iterable[Iterable[Hashable]],
        *,
        universe: Iterable[Hashable] | None = None,
    ) -> "Partition":
        raw_blocks = [tuple(block) for block in blocks]
        if not raw_blocks:
            raise PartitionError("a partition must contain at least one block")
        if any(len(block) == 0 for block in raw_blocks):
            raise PartitionError("partition blocks must be non-empty")

        seen: set[Hashable] = set()
        for block in raw_blocks:
            for state in block:
                if state in seen:
                    raise PartitionError(
                        f"state appears in more than one block: {state!r}"
                    )
                seen.add(state)

        if universe is not None:
            ordered_universe = tuple(universe)
            universe_set = set(ordered_universe)
            if seen != universe_set:
                missing = universe_set - seen
                extra = seen - universe_set
                raise PartitionError(
                    f"blocks must cover the universe exactly; missing={missing!r}, extra={extra!r}"
                )
            order = {state: i for i, state in enumerate(ordered_universe)}
            raw_blocks = [
                tuple(sorted(block, key=lambda state: order[state]))
                for block in raw_blocks
            ]
            raw_blocks.sort(key=lambda block: min(order[state] for state in block))

        return cls(tuple(raw_blocks))

    @classmethod
    def discrete(cls, states: Iterable[Hashable]) -> "Partition":
        return cls.from_blocks(((state,) for state in states), universe=states)

    @classmethod
    def from_mapping(
        cls,
        mapping: Mapping[Hashable, Hashable],
        *,
        universe: Iterable[Hashable],
    ) -> "Partition":
        blocks_by_value: dict[Hashable, list[Hashable]] = {}
        for state in universe:
            blocks_by_value.setdefault(mapping[state], []).append(state)
        return cls.from_blocks(blocks_by_value.values(), universe=universe)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Partition):
            return NotImplemented
        return self._block_set == other._block_set

    def __hash__(self) -> int:
        return hash(self._block_set)

    def __len__(self) -> int:
        return len(self.blocks)

    def __iter__(self):
        return iter(self.blocks)

    @property
    def states(self) -> tuple[Hashable, ...]:
        return tuple(state for block in self.blocks for state in block)

    @property
    def _block_set(self) -> frozenset[frozenset[Hashable]]:
        return frozenset(frozenset(block) for block in self.blocks)

    def block_index(self, state: Hashable) -> int:
        for i, block in enumerate(self.blocks):
            if state in block:
                return i
        raise KeyError(state)

    def block_for(self, state: Hashable) -> tuple[Hashable, ...]:
        return self.blocks[self.block_index(state)]

    def same_block(self, left: Hashable, right: Hashable) -> bool:
        return self.block_index(left) == self.block_index(right)

    def as_mapping(self) -> dict[Hashable, int]:
        return {state: i for i, block in enumerate(self.blocks) for state in block}

    def refines(self, other: "Partition") -> bool:
        """Return True if every block of self is contained in a block of other."""

        other_blocks = [set(block) for block in other.blocks]
        return all(
            any(set(block) <= other_block for other_block in other_blocks)
            for block in self.blocks
        )

    def is_strictly_finer_than(self, other: "Partition") -> bool:
        return self != other and self.refines(other)

    def is_support_over(self, public: Mapping[Hashable, Hashable]) -> bool:
        """Return True when the public projection factors through this partition."""

        for block in self.blocks:
            public_values = {public[state] for state in block}
            if len(public_values) > 1:
                return False
        return True

    def format(self) -> str:
        block_text = ["{" + ", ".join(map(str, block)) + "}" for block in self.blocks]
        return "{" + ", ".join(block_text) + "}"


def common_coarsening(
    partitions: Iterable[Partition], *, universe: Iterable[Hashable]
) -> Partition:
    """Return the common coarsening generated by the partitions' equivalence relations."""

    ordered_universe = tuple(universe)
    parent = {state: state for state in ordered_universe}

    def find(state: Hashable) -> Hashable:
        while parent[state] != state:
            parent[state] = parent[parent[state]]
            state = parent[state]
        return state

    def union(left: Hashable, right: Hashable) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    saw_partition = False
    for partition in partitions:
        saw_partition = True
        for block in partition.blocks:
            first = block[0]
            for state in block[1:]:
                union(first, state)
    if not saw_partition:
        raise PartitionError("at least one partition is required")

    groups: dict[Hashable, list[Hashable]] = {}
    for state in ordered_universe:
        groups.setdefault(find(state), []).append(state)
    return Partition.from_blocks(groups.values(), universe=ordered_universe)


def all_partitions(items: Iterable[Hashable]) -> list[tuple[tuple[Hashable, ...], ...]]:
    """Enumerate all set partitions of items in a stable order."""

    ordered_items = tuple(items)
    if not ordered_items:
        return [()]

    first = ordered_items[0]
    rest_partitions = all_partitions(ordered_items[1:])
    out: list[tuple[tuple[Hashable, ...], ...]] = []
    for partition in rest_partitions:
        out.append(((first,),) + partition)
        for i, block in enumerate(partition):
            merged = block + (first,)
            out.append(partition[:i] + (merged,) + partition[i + 1 :])
    return out


def partitions_refining_public(
    states: Iterable[Hashable],
    public: Mapping[Hashable, Hashable],
) -> list[Partition]:
    """Enumerate all supports over a public projection."""

    ordered_states = tuple(states)
    fibers: dict[Hashable, list[Hashable]] = {}
    for state in ordered_states:
        fibers.setdefault(public[state], []).append(state)

    per_fiber_partitions = [all_partitions(fiber) for fiber in fibers.values()]
    partitions: list[Partition] = []
    for fiber_product in product(*per_fiber_partitions):
        blocks = [
            block for fiber_partition in fiber_product for block in fiber_partition
        ]
        partitions.append(Partition.from_blocks(blocks, universe=ordered_states))
    return partitions
