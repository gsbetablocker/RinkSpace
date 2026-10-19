"""Guard the user-confirmed advertisement identities across scene revisions."""

from __future__ import annotations


def validate_user_confirmed_advertisements(scene: dict, contract: dict, contract_sha256: str) -> None:
    """Check immutable semantic relations, while allowing later physical fitting."""
    if contract.get("schema_version") != "rinkspace.user-confirmed-advertisement-topology.v1":
        raise ValueError("invalid user-confirmed advertisement contract schema")
    topology = scene.get("advertisement_topology")
    if not isinstance(topology, dict) or topology.get("contract_sha256") != contract_sha256:
        raise ValueError("scene lost the user-confirmed advertisement contract")
    if topology.get("status") != "user_confirmed_hard_semantic_relationships":
        raise ValueError("scene downgraded user-confirmed advertisement relationships")
    for field in ("perimeter_order_counterclockwise_from_upper_right", "side_sequences",
                  "same_content_pairs", "schematic_central_symmetry"):
        if topology.get(field) != contract.get(field):
            raise ValueError(f"scene changed confirmed advertisement {field}")

    positions = contract.get("positions", [])
    adverts = scene.get("advertisements", [])
    if len(positions) != 26 or len(adverts) != 26:
        raise ValueError("confirmed advertisement ring must contain 26 physical instances")
    if [p.get("number") for p in positions] != list(range(1, 27)):
        raise ValueError("advertisement contract numbering is incomplete")
    by_number = {ad.get("user_number"): ad for ad in adverts}
    if len(by_number) != 26 or set(by_number) != set(range(1, 27)):
        raise ValueError("scene advertisement numbering is incomplete or duplicated")
    if len({ad.get("id") for ad in adverts}) != 26:
        raise ValueError("opposite advertisements must remain separate physical instances")
    pairs = {n: pair for pair in contract["same_content_pairs"] for n in pair["numbers"]}
    if len(pairs) != 26:
        raise ValueError("confirmed same-content pairs do not cover the ring")
    for position in positions:
        number = position["number"]
        ad = by_number[number]
        partner_number = number + 13 if number <= 13 else number - 13
        expected = {
            "id": position["instance_id"],
            "review_ppt_label": position["ppt_label"],
            "qualitative_side": position["region"],
            "perimeter_order_counterclockwise": number,
            "schematic_anchor_shape_name": position["ppt_anchor_shape_name"],
            "schematic_anchor_center_emu": position["schematic_anchor_center_emu"],
            "visual_content_id": pairs[number]["content_id"],
            "same_content_partner_number": partner_number,
            "same_content_partner_id": by_number[partner_number]["id"],
        }
        for field, value in expected.items():
            if ad.get(field) != value:
                raise ValueError(f"advertisement {number} changed confirmed {field}")
        if ad.get("use_as_ice_point") is not False:
            raise ValueError(f"advertisement {number} cannot become an ice point")
    if topology.get("number_to_instance_id") != {
        str(n): by_number[n]["id"] for n in range(1, 27)
    }:
        raise ValueError("number-to-instance identity map changed")
