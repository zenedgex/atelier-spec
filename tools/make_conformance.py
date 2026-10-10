"""Write the bad cases of the conformance suite: a valid base contract or machine, one change each.

Each file starts with `# expect: <text>`; a conforming checker reports an error containing that text.
Run: python tools/make_conformance.py   (rewrites conformance/*/bad/)
"""
import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
BLOCK = yaml.safe_load((ROOT / "conformance/block/good/acme_vec.block.yaml").read_text())
MACHINE = yaml.safe_load((ROOT / "conformance/machine/good/npu.yaml").read_text())


def cmd(s):
    return s["programming"]["commands"][0]


def field(s, name):
    return next(f for f in cmd(s)["fields"] if f["name"] == name)


BLOCK_CASES = {
    "field_outside_word": (lambda s: field(s, "shift").update(bits=[120, 130]), "do not fit a 128-bit word"),
    "overlapping_fields": (lambda s: field(s, "src_b").update(bits=[30, 55]), "is also in src_a"),
    "unknown_op": (lambda s: cmd(s)["semantics"][0].update(op="fma"), "op 'fma' is not in the op set"),
    "undefined_input": (lambda s: cmd(s)["semantics"][1].update(**{"in": ["u"]}), "input 'u' is neither"),
    "cycles_unknown_name": (lambda s: cmd(s).update(cycles="4 + len / WIDTH"), "['WIDTH'] are neither"),
    "duplicate_opcode": (lambda s: s["programming"]["commands"][1].update(opcode=0x12), "opcode 18 used twice"),
    "opcode_too_wide": (lambda s: cmd(s).update(opcode=0x1FF), "does not fit bits [0, 7]"),
    "accelerator_without_commands": (lambda s: (s.update(slot_class="accelerator.op"),
                                                s["programming"].clear(), s["programming"].update(model="none")),
                                     "an accelerator.op block is programmed by commands"),
    "unknown_protocol": (lambda s: s["port_groups"][1].update(protocol="wishbone"), "unknown protocol 'wishbone'"),
    "missing_role": (lambda s: s["port_groups"][0].update(ports={"rst_n": "rst_ni"}), "missing roles ['clk']"),
    "port_in_two_groups": (lambda s: s["port_groups"][3].update(ports={"irq": "clk_i"}), "port clk_i is also in group clk"),
    "bad_evidence": (lambda s: s["cost"]["area"].update(evidence="guess"), "evidence 'guess' is not a label"),
    "tensor_without_kind": (lambda s: s.update(slot_class="compute.tensor"), "needs kind"),
    "own_op_without_namespace": (lambda s: s["programming"].update(ops={"fft": {"reference": "model.py"}}),
                                 "needs a namespace"),
    "own_op_without_reference": (lambda s: s["programming"].update(ops={"acme.fft": {}}), "needs a reference"),
    "unknown_slot_class": (lambda s: s.update(slot_class="compute.quantum"), "is not a slot class"),
}

MACHINE_CASES = {
    "address_overlap": (lambda m: m["address_map"]["l2"].update(base=0x8000), "overlaps"),
    "cluster_memory_outside_cluster": (lambda m: m["clusters"]["engine"]["contains"].remove("acc"),
                                       "per: cluster, but no cluster contains it"),
    "core_is_not_a_core": (lambda m: m["control"].update(core="dma"), "'dma' is a movement.dma"),
    "control_core_missing": (lambda m: m["control"].update(core="cpu9"), "'cpu9' is not a unit"),
    "unknown_slave": (lambda m: m["interconnect"]["slaves"].append("l3"), "'l3' names nothing"),
    "space_unknown_path": (lambda m: m["space"].update({"memory.l3.size": ["1MiB"]}), "space.memory.l3.size"),
    "duplicate_id": (lambda m: m["memory"].append({"name": "dma", "level": 2, "kind": "scratchpad", "per": "chip",
                                                   "size": "1MiB"}), "id also used by a unit"),
    "two_clock_domains": (lambda m: m["clocks"]["engine"]["units"].append("core"), "also in clock domain core"),
    "banks_do_not_divide": (lambda m: m["memory"][0].update(banks=3), "does not split into 3 banks"),
    "dma_to_unit": (lambda m: m["movement"]["dma"][0].update(to="tensor"), "to: 'tensor' is not a memory"),
    "unknown_class": (lambda m: m["units"]["vec"].update({"class": "compute.vectorz"}), "is not a slot class"),
    # 0.2: dispatch, DMA engines, stream ports, port sharing (spec 16)
    "dispatch_by_unknown": (lambda m: m["dispatch"].update(by="disp9"), "'disp9' is not a unit"),
    "dispatch_by_a_tensor_unit": (lambda m: m["dispatch"].update(by="tensor"), "dispatch is a control.dispatcher"),
    "dispatch_per_unknown": (lambda m: m["dispatch"].update(per="pod"), "'pod' is not chip or a cluster id"),
    "queue_for_a_sequencer": (lambda m: m["dispatch"].update(queues={"seq": {"depth": 4}}), "'seq' takes no queue"),
    "no_tokens": (lambda m: m["dispatch"].update(tokens=0), "less than the minimum of 1"),
    "dma_engine_not_a_dma": (lambda m: m["movement"]["dma"][0].update(engine="vec"), "not a movement.dma"),
    "stream_of_no_unit": (lambda m: m["movement"].update(streams=[{"name": "cam", "unit": "cam9", "direction": "in"}]),
                          "unit 'cam9' is not a unit"),
    "unknown_share": (lambda m: m["memory"][0].update(share="triple"), "'triple' is not one of"),
}


def write(base, cases, out):
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*.yaml"):
        f.unlink()
    for name, (change, expect) in cases.items():
        s = copy.deepcopy(base)
        change(s)
        (out / f"{name}.yaml").write_text(f"# expect: {expect}\n" + yaml.safe_dump(s, sort_keys=False))


if __name__ == "__main__":
    write(BLOCK, BLOCK_CASES, ROOT / "conformance/block/bad")
    write(MACHINE, MACHINE_CASES, ROOT / "conformance/machine/bad")
    print(len(BLOCK_CASES), "block cases,", len(MACHINE_CASES), "machine cases")
