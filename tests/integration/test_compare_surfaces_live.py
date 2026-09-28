"""compare_surfaces on long slivers: a closest point, not a nearest centroid.

Executors' own fit scripts picked candidate triangles by nearest centroid and
read 111 mm on a CAD rod whose triangles ran 818 mm between two rings: the
centroid of the right face was too far away to be a candidate. That was
blamed on compare_surfaces, which measures with the Ray SOP's minimum
distance and was never wrong here (checked against brute force, 2026-09-28).
This pins it: the same rod, coarse against a finely ringed copy 0.3 mm
larger, reads 0.3 mm at most in both directions.
"""

from __future__ import annotations

# Third-party
import pytest

pytestmark = pytest.mark.integration

_LENGTH = 0.818
_RADIUS = 0.006
_OFFSET = 0.0003


@pytest.fixture
def rods(call) -> tuple[str, str]:
    geo = call("nodes.create_node", parent_path="/obj", node_type="geo", name="rods")
    paths = []
    for name, rows, radius in (("coarse", 2, _RADIUS), ("fine", 410, _RADIUS + _OFFSET)):
        node = call("nodes.create_node", parent_path=geo["node_path"], node_type="tube", name=name)
        call(
            "parameters.set_parameters",
            node_path=node["node_path"],
            params={
                "type": "poly",
                "rows": rows,
                "cols": 12,
                "rad": [radius, radius],
                "height": _LENGTH,
                "cap": 0,
            },
        )
        paths.append(node["node_path"])
    return paths[0], paths[1]


def test_long_slivers_read_their_true_distance_both_ways(call, rods):
    coarse, fine = rods
    data = call("modeling.compare_surfaces", a=coarse, b=fine, samples=4000)
    # Parallel chords 0.3 mm apart radially are 0.3 * cos(15 deg) = 0.29 mm
    # apart; at the ring vertices the gap is the full 0.3 mm.
    for direction in ("a_to_b", "b_to_a"):
        assert data[direction]["max"] <= _OFFSET * 1.001, (direction, data[direction])
        assert data[direction]["mean"] >= _OFFSET * 0.95, (direction, data[direction])
