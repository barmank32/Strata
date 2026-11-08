"""OUR FORK (video-input): STRATA_DFLASH2_TUNED runs the tuned DFlash2 pack on a card the entry's class does not list.

Upstream gates the tuned pack by card class: for CUDA the set's entry says ``where="resident"``, which is the
CONF_STEPS step that gives ``--draft-conf 0.35`` - 12,000 engine slots (drafting.tuned_applies). The stand this fork
runs is one CUDA sm_70 card with ``--expert-cache auto``, whose slots_of() reads 11,869: 131 slots under the step, so
upstream would never take the pack there and nobody has measured the cell. The variable widens WHERE the measured
default runs and nothing else - the tuned file is the same format and the same 339,694,592 bytes as the converted
stock pack (drafting.TUNED_NAME, drafting.CONVERTED_NAME), so the engine, its VRAM figure and the confidence rule are
untouched, and a card without the verified bytes still takes the stock pack because pack_path asks tuned_ok. Unset,
empty or 0 answers exactly as upstream, and a card class that has its own rule for the set (CLASS_DEFAULTS) wins over
the variable.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from strata_control import drafting

IQ3S = "qwen3.8-flash-next-iq3_s"
Q4KXL = "qwen3.8-flash-next-unsloth-ud-q4_k_xl"

#: the stand's own arguments, read out of its run config on 2026-11-08 (the pack, the auto expert cache, the KV terms)
STAND_ARGS = ["--pack", "/opt/Strata-data/packs/iq3_s", "--expert-cache", "auto", "--expert-cache-per-layer",
              "--kv-resident", "32768", "--kv", "k8v4", "--max-context", "262144", "--vram-reserve-mib", "1024",
              "--mtp", "/opt/Strata-data/mtp/rt"]
STAND_VRAM_MIB = 32768           # the one card's memory, as serve/server.py drafting_vram_mib gives it
STAND_ARCHS = ["sm_70"]


class EnvOverride(unittest.TestCase):
    def _applies(self, env: str | None, **kw) -> bool:
        with mock.patch.dict(os.environ, {}, clear=False):
            if env is None:
                os.environ.pop("STRATA_DFLASH2_TUNED", None)
            else:
                os.environ["STRATA_DFLASH2_TUNED"] = env
            return drafting.tuned_applies(**kw)

    def test_the_stand_is_under_the_step_without_the_variable(self) -> None:
        """Unchanged upstream answer: 11,869 estimated slots, the step is 12,000."""
        self.assertEqual(drafting.slots_of(list(STAND_ARGS), STAND_VRAM_MIB), 11869)
        self.assertLess(drafting.slots_of(list(STAND_ARGS), STAND_VRAM_MIB), drafting.CONF_STEPS[0][0])
        self.assertFalse(self._applies(None, backend="cuda", model_name=IQ3S, archs=STAND_ARCHS,
                                       args=list(STAND_ARGS), vram_mib=STAND_VRAM_MIB))

    def test_the_variable_widens_the_class(self) -> None:
        for yes in ("1", "on", "YES", "true"):
            self.assertTrue(self._applies(yes, backend="cuda", model_name=IQ3S, archs=STAND_ARCHS,
                                          args=list(STAND_ARGS), vram_mib=STAND_VRAM_MIB), yes)

    def test_the_variable_off_and_empty_keep_the_upstream_answer(self) -> None:
        for no in (None, "", "0", "auto"):
            self.assertFalse(self._applies(no, backend="cuda", model_name=IQ3S, archs=STAND_ARCHS,
                                           args=list(STAND_ARGS), vram_mib=STAND_VRAM_MIB), repr(no))

    def test_a_set_without_a_tuned_entry_stays_without_one(self) -> None:
        """CUDA UD-Q4_K_XL has no tuned entry upstream, and the variable does not invent one."""
        self.assertIsNone(drafting.tuned_entry("cuda", Q4KXL))
        self.assertFalse(self._applies("1", backend="cuda", model_name=Q4KXL, archs=STAND_ARCHS,
                                       args=list(STAND_ARGS), vram_mib=STAND_VRAM_MIB))

    def test_a_card_class_with_its_own_rule_still_wins(self) -> None:
        """The 12 GB / streaming class sends IQ3_S to the MTP; the variable does not put it back on DFlash2."""
        small = ["--expert-cache", "1500"]        # 1500 x 1.3 = 1950 engine slots: under the 3000 step
        self.assertTrue(drafting.streaming_class("cuda", ["sm_70"], args=small, vram_mib=None))
        self.assertFalse(self._applies("1", backend="cuda", model_name=IQ3S, archs=["sm_70"], args=small,
                                       vram_mib=None))

    def test_the_pack_choice_still_needs_the_verified_bytes(self) -> None:
        """With the variable on and no drafter file on disk, pack_path stays on the stock pack."""
        with tempfile.TemporaryDirectory() as data:
            with mock.patch.dict(os.environ, {"STRATA_DFLASH2_TUNED": "1"}, clear=False):
                pack = drafting.pack_path(data, engine=None, model_name=IQ3S, backend="cuda")
            self.assertEqual(Path(pack).name, drafting.PACK_NAME)
            self.assertNotEqual(Path(pack).name, drafting.TUNED_NAME)


if __name__ == "__main__":
    unittest.main()
