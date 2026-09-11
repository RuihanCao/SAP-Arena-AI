"""Two pieces:

1. `build_observation_layout(mode, max_turn)` -- reconstructs the v4 layout's
   segment boundaries as a plain-int dict, READING every size off a
   constructed `StateVectorEncoderV4` instance (`observation.py`'s
   `build_state_encoder`, the exact same factory `chain_bc_dataset.py` uses
   to build the cache), the same way `python/tests/test_observation.py`
   already does for its own slicing assertions (e.g. `enc._team_slot_size`,
   `enc._pet_ids`) -- never a hardcoded magic number for anything the encoder
   itself exposes. A handful of sizes have no corresponding encoder
   attribute (the per-slot NUMERIC feature counts, and the opponent-context
   sub-block widths) because they are inline literals inside
   `observation.py`/`tempo/features.py`'s own encode functions; those are
   pinned here as named module constants with exact source pointers, and
   every returned number is cross-checked against the encoder's own
   aggregate sizes (`_team_slot_size`, `_shop_slot_size`,
   `_opponent_context_size`, `.size`), raising `ValueError` loudly the
   moment any of those constants drifts out of sync with `observation.py`.
   This dict is what `SlotEmbeddingExtractor` uses to slice the flat 1993-dim
   vector; the two modules are meant to be used together, never independently
   re-implemented.

2. `SlotEmbeddingExtractor` -- a `BaseFeaturesExtractor` that:
   - Slices the flat observation ONCE (in `__init__`, into plain python int
     offsets) into: global(6), 5 team slots(121 each), 9 shop slots(108
     each), summary-base(20) + opponent-context(390).
   - Runs a SHARED `nn.Linear` pet-identity embedding (`pet_core`) over BOTH
     the team side (68-wide one-hot with an explicit empty bucket -- handled
     via a separate learned `pet_empty` vector, since "no pet" is not itself
     a species) and the shop side (67-wide one-hot, no empty bucket -- a
     non-pet/empty shop slot naturally embeds to zero). One weight matrix
     covers every pet everywhere on the board, which is the entire point:
     the model can now see that two different board positions hold "the same
     species" instead of two unrelated orthogonal basis vectors.
   - Runs a SHARED per-team-slot MLP over all 5 team slots, and a SHARED
     per-shop-slot MLP over all 9 shop slots (weight sharing falls out for
     free from applying one `nn.Linear`/`nn.Sequential` to a reshaped
     `[B, n_slots, in_dim]` tensor -- no explicit loop needed).
   - `arch="slot_v1"`: concatenates [global+summary-base block, 5 team-slot
     vectors in board-position order, 9 shop-slot vectors in position order,
     opponent block] -- position order is kept (not set-pooled) because SAP
     board position carries real game semantics (adjacency for triggers,
     "front of the line" for combat order).
   - `arch="slot_attn"`: same slot vectors, additionally passed through one
     shared, position-embedded self-attention layer before the same concat,
     letting slots attend to each other (e.g. "is there a pair of this pet
     elsewhere on my board") before the final MLP trunk.

Both modules are v4-only by construction (the opponent-context block only
exists on `StateVectorEncoderV4`); `build_observation_layout` raises loudly
if asked to build a layout for any other observation mode, rather than
silently returning a layout that doesn't describe what `SlotEmbeddingExtractor`
assumes."""

from __future__ import annotations

import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from .observation import build_state_encoder

# ---------------------------------------------------------------------------
# Fixed architectural constants that have NO corresponding encoder attribute
# to derive from (they are inline literals inside observation.py / a helper
# it calls) -- see the exact source lines in each comment. Never scattered
# inline; always referenced from here so a drift in the source formula shows
# up as a single loud ValueError from `build_observation_layout` instead of a
# silent mis-slice inside `SlotEmbeddingExtractor.forward`.
# ---------------------------------------------------------------------------

# observation.py's SAP board-size constants (`_MAX_TEAM_SLOTS`/`_MAX_SHOP_SLOTS`,
# both module-private there). Re-declared locally rather than imported, matching
# this codebase's own precedent for cross-module reuse of these two numbers
# (`tempo/features.py`'s `_V2_MAX_TEAM_SLOTS = 5`) -- these are SAP game-engine
# constants (5 team slots, 9 shop slots), not tunable knobs.
_N_TEAM_SLOTS = 5
_N_SHOP_SLOTS = 9

# observation.py::StateVectorEncoderV2._team_slot's numeric/combine/trigger
# tail (`out.extend([attack, health, perm_attack, perm_health, temp_attack,
# temp_health, level, exp, sell_value, pet_tier, combine_team, combine_shop,
# trigger_limit, trigger_consumed, trigger_left])`, ~lines 481-503).
_TEAM_NUMERIC_COUNT = 15

# observation.py::StateVectorEncoderV2._shop_slot's `slot_type_onehot`
# (`[0.0, 0.0]`, ~line 518) and numeric tail (`out.extend([cost, frozen,
# pet_attack, pet_health, shop_bonus_at, shop_bonus_hp, item_tier, is_linked,
# linked_partner_exists, link_group_size, buy_combine_target_count])`,
# ~lines 553-567).
_SHOP_TYPE_ONEHOT_SIZE = 2
_SHOP_NUMERIC_COUNT = 11

# tempo/features.py::_team_summary's fixed-width vector (pet_count,
# total_attack, total_health, total_power, avg_level) and
# ::encode_last_opponent_context's appended `opp_lives` scalar.
_OPPONENT_SUMMARY_COUNT = 5
_OPPONENT_LIVES_COUNT = 1

# `--features` choices that route through this module (excludes "flat",
# which means "don't build a layout / don't attach this extractor at all" --
# see train_chain_bc.py::_build_model).
SLOT_ARCH_CHOICES = ("slot_v1", "slot_attn")


def build_observation_layout(mode: str, max_turn: int) -> dict[str, int]:
    """Reconstruct the v4 observation's segment boundaries as a plain-int
    dict (pickle-safe -- this is stored directly in sb3 `policy_kwargs`).

    Every size that the encoder itself tracks is READ off a constructed
    encoder instance, never hardcoded (`enc._global_size`,
    `enc._team_slot_size`, `enc._shop_slot_size`, `enc._summary_size`,
    `len(enc._pet_ids)`, `len(enc._equip_ids)`, `len(enc._status_ids)`,
    `enc._opponent_context_size`, `enc.opponent_hash_buckets`), the same
    precedent `test_observation.py` already establishes for slicing
    assertions. A few sizes have no encoder attribute at all (see the
    `_TEAM_NUMERIC_COUNT` etc. constants above); those are cross-checked
    against the encoder's own aggregate sizes below, and any mismatch raises
    `ValueError` loudly -- this is the ONE place a future change to
    `observation.py`'s per-slot layout would need to also update, and this
    function's job is to make forgetting that impossible to miss silently.

    v4-only: raises `ValueError` for any other observation mode, since the
    fixed team/shop/opponent segment structure `SlotEmbeddingExtractor`
    relies on (in particular the opponent-context block) only exists on
    `StateVectorEncoderV4`.
    """
    encoder = build_state_encoder(observation_mode=mode, max_turn=int(max_turn))
    opponent_hash_buckets = getattr(encoder, "opponent_hash_buckets", None)
    opponent_context_size = getattr(encoder, "_opponent_context_size", None)
    if opponent_hash_buckets is None or opponent_context_size is None:
        raise ValueError(
            "build_observation_layout_requires_v4:"
            f"mode={mode!r}:encoder_class={type(encoder).__name__}:"
            "SlotEmbeddingExtractor's fixed team/shop/opponent segment sizes "
            "assume the v4_one_turn_context encoder's opponent-hash-context "
            "block; no other observation mode is supported"
        )

    total = int(encoder.size)
    global_size = int(encoder._global_size)
    team_slot_size = int(encoder._team_slot_size)
    shop_slot_size = int(encoder._shop_slot_size)
    summary_size = int(encoder._summary_size)

    computed_total = global_size + (_N_TEAM_SLOTS * team_slot_size) + (_N_SHOP_SLOTS * shop_slot_size) + summary_size
    if computed_total != total:
        raise ValueError(
            "observation_layout_total_mismatch:"
            f"global={global_size}+{_N_TEAM_SLOTS}*team={team_slot_size}"
            f"+{_N_SHOP_SLOTS}*shop={shop_slot_size}+summary={summary_size}"
            f"={computed_total}!=encoder.size={total}"
        )

    team_pet = 1 + len(encoder._pet_ids)  # empty bucket + one column per species
    team_equip = 1 + len(encoder._equip_ids)  # none bucket + one column per equip
    team_status = len(encoder._status_ids)
    team_num = _TEAM_NUMERIC_COUNT
    computed_team = team_pet + team_equip + team_status + team_num
    if computed_team != team_slot_size:
        raise ValueError(
            "observation_layout_team_slot_mismatch:"
            f"pet={team_pet}+equip={team_equip}+status={team_status}+num={team_num}"
            f"={computed_team}!=encoder._team_slot_size={team_slot_size}"
        )

    shop_type = _SHOP_TYPE_ONEHOT_SIZE
    shop_pet = len(encoder._pet_ids)  # no empty bucket on the shop side
    shop_food = len(encoder._food_ids)
    shop_num = _SHOP_NUMERIC_COUNT
    computed_shop = shop_type + shop_pet + shop_food + shop_num
    if computed_shop != shop_slot_size:
        raise ValueError(
            "observation_layout_shop_slot_mismatch:"
            f"type={shop_type}+pet={shop_pet}+food={shop_food}+num={shop_num}"
            f"={computed_shop}!=encoder._shop_slot_size={shop_slot_size}"
        )

    opp_summary = _OPPONENT_SUMMARY_COUNT
    opp_hash = 3 * int(opponent_hash_buckets)
    opp_lives = _OPPONENT_LIVES_COUNT
    opp_size = opp_summary + opp_hash + opp_lives
    if opp_size != int(opponent_context_size):
        raise ValueError(
            "observation_layout_opponent_block_mismatch:"
            f"summary={opp_summary}+hash={opp_hash}+lives={opp_lives}"
            f"={opp_size}!=encoder._opponent_context_size={int(opponent_context_size)}"
        )

    summary_base = summary_size - opp_size
    if summary_base < 0:
        raise ValueError(f"observation_layout_summary_base_negative:summary_size={summary_size}<opp_size={opp_size}")

    layout = {
        "total": total,
        "global_size": global_size,
        "n_team": _N_TEAM_SLOTS,
        "team_slot_size": team_slot_size,
        "team_pet": team_pet,
        "team_equip": team_equip,
        "team_status": team_status,
        "team_num": team_num,
        "n_shop": _N_SHOP_SLOTS,
        "shop_slot_size": shop_slot_size,
        "shop_type": shop_type,
        "shop_pet": shop_pet,
        "shop_food": shop_food,
        "shop_num": shop_num,
        "summary_size": summary_size,
        "summary_base": summary_base,
        "opp_size": opp_size,
        "opp_summary": opp_summary,
        "opp_hash": opp_hash,
        "opp_lives": opp_lives,
    }
    return {str(k): int(v) for k, v in layout.items()}


class SlotEmbeddingExtractor(BaseFeaturesExtractor):
    """`layout` must come from `build_observation_layout` (or something with
    the exact same keys/invariants) -- this class trusts its internal
    consistency and only re-checks that `observation_space` actually matches
    `layout["total"]`."""

    def __init__(
        self,
        observation_space: spaces.Box,
        layout: dict[str, int],
        d_pet: int = 32,
        d_item: int = 16,
        d_slot: int = 48,
        d_opp: int = 32,
        arch: str = "slot_v1",
        attn_heads: int = 2,
    ) -> None:
        if arch not in SLOT_ARCH_CHOICES:
            raise ValueError(f"unknown_slot_embedding_arch:{arch!r}:allowed={SLOT_ARCH_CHOICES}")
        expected_shape = (int(layout["total"]),)
        if tuple(observation_space.shape) != expected_shape:
            raise ValueError(
                "slot_embedding_extractor_observation_shape_mismatch:"
                f"got={tuple(observation_space.shape)}:expected={expected_shape}"
            )

        n_team = int(layout["n_team"])
        n_shop = int(layout["n_shop"])
        d_slot = int(d_slot)
        d_opp = int(d_opp)
        features_dim = 32 + ((n_team + n_shop) * d_slot) + d_opp
        super().__init__(observation_space, features_dim=features_dim)

        self.layout = dict(layout)
        self.d_pet = int(d_pet)
        self.d_item = int(d_item)
        self.d_slot = d_slot
        self.d_opp = d_opp
        self.arch = str(arch)
        self.attn_heads = int(attn_heads)
        self._n_team = n_team
        self._n_shop = n_shop

        # ---- slicing offsets: computed ONCE here, stored as plain python
        # ints (never tensors/buffers -- these are pure indexing metadata,
        # not learnable state); forward() only ever slices with them. ----
        global_size = int(layout["global_size"])
        team_slot_size = int(layout["team_slot_size"])
        shop_slot_size = int(layout["shop_slot_size"])
        summary_size = int(layout["summary_size"])
        team_pet, team_equip, team_status, team_num = (
            int(layout["team_pet"]),
            int(layout["team_equip"]),
            int(layout["team_status"]),
            int(layout["team_num"]),
        )
        shop_type, shop_pet, shop_food, shop_num = (
            int(layout["shop_type"]),
            int(layout["shop_pet"]),
            int(layout["shop_food"]),
            int(layout["shop_num"]),
        )
        summary_base = int(layout["summary_base"])
        opp_size = int(layout["opp_size"])

        self._team_slot_size = team_slot_size
        self._shop_slot_size = shop_slot_size
        self._global_start, self._global_end = 0, global_size
        self._team_start = self._global_end
        self._team_end = self._team_start + (n_team * team_slot_size)
        self._shop_start = self._team_end
        self._shop_end = self._shop_start + (n_shop * shop_slot_size)
        self._summary_start = self._shop_end
        self._summary_end = self._summary_start + summary_size
        if self._summary_end != int(layout["total"]):
            raise ValueError(
                "slot_embedding_extractor_layout_does_not_tile_observation:"
                f"summary_end={self._summary_end}!=total={layout['total']}"
            )


        self._t_pet0, self._t_pet1 = 0, team_pet
        self._t_equip0, self._t_equip1 = self._t_pet1, self._t_pet1 + team_equip
        self._t_status0, self._t_status1 = self._t_equip1, self._t_equip1 + team_status
        self._t_num0, self._t_num1 = self._t_status1, self._t_status1 + team_num
        # within-shop-slot offsets: type(2) -> pet(67) -> food(28) -> num(11)
        self._s_type0, self._s_type1 = 0, shop_type
        self._s_pet0, self._s_pet1 = self._s_type1, self._s_type1 + shop_pet
        self._s_food0, self._s_food1 = self._s_pet1, self._s_pet1 + shop_food
        self._s_num0, self._s_num1 = self._s_food1, self._s_food1 + shop_num
        # within-summary offsets: base(summary_size-390) -> opponent(390)
        self._sum_base0, self._sum_base1 = 0, summary_base
        self._sum_opp0, self._sum_opp1 = summary_base, summary_base + opp_size

        # ---- modules ----
        # Shared pet-identity embedding: ONE weight matrix for every pet on
        # the board, team or shop (see module docstring). `bias=False` so a
        # true zero one-hot (shop slot holding no pet) embeds to exactly
        # zero, not a learned constant.
        self.pet_core = nn.Linear(shop_pet, self.d_pet, bias=False)
        # Team-only: the empty-bucket contribution is a SEPARATE learned
        # vector (an empty slot is not "a pet species", so it must not share
        # pet_core's rows).
        self.pet_empty = nn.Parameter(th.zeros(self.d_pet))
        self.team_equip_proj = nn.Linear(team_equip, self.d_item, bias=False)
        self.shop_food_proj = nn.Linear(shop_food, self.d_item, bias=False)

        team_slot_in_dim = self.d_pet + self.d_item + team_status + team_num
        self.team_slot_encoder = nn.Sequential(nn.Linear(team_slot_in_dim, self.d_slot), nn.ReLU())
        shop_slot_in_dim = shop_type + self.d_pet + self.d_item + shop_num
        self.shop_slot_encoder = nn.Sequential(nn.Linear(shop_slot_in_dim, self.d_slot), nn.ReLU())

        self.opp_encoder = nn.Sequential(nn.Linear(opp_size, self.d_opp), nn.ReLU())
        self.global_encoder = nn.Sequential(nn.Linear(global_size + summary_base, 32), nn.ReLU())

        self.pos_embed: nn.Parameter | None = None
        self.slot_attn: nn.MultiheadAttention | None = None
        if self.arch == "slot_attn":
            self.pos_embed = nn.Parameter(th.zeros(n_team + n_shop, self.d_slot))
            self.slot_attn = nn.MultiheadAttention(self.d_slot, self.attn_heads, batch_first=True)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        x = observations
        batch = int(x.shape[0])

        global_seg = x[:, self._global_start : self._global_end]
        team_block = x[:, self._team_start : self._team_end].reshape(batch, self._n_team, self._team_slot_size)
        shop_block = x[:, self._shop_start : self._shop_end].reshape(batch, self._n_shop, self._shop_slot_size)
        summary_block = x[:, self._summary_start : self._summary_end]
        summary_base_seg = summary_block[:, self._sum_base0 : self._sum_base1]
        opp_seg = summary_block[:, self._sum_opp0 : self._sum_opp1]

        team_pet_onehot = team_block[..., self._t_pet0 : self._t_pet1]
        team_equip_onehot = team_block[..., self._t_equip0 : self._t_equip1]
        team_status = team_block[..., self._t_status0 : self._t_status1]
        team_num = team_block[..., self._t_num0 : self._t_num1]

        # empty bucket (index 0) -> learned pet_empty vector; real pet
        # (indices 1..) -> pet_core over the 67-wide one-hot tail. Exactly
        # one of the two is active per slot (one-hot), so this is a clean
        # select, not a blend.
        team_pet_vec = self.pet_core(team_pet_onehot[..., 1:]) + team_pet_onehot[..., :1] * self.pet_empty
        team_equip_vec = self.team_equip_proj(team_equip_onehot)
        team_slot_in = th.cat([team_pet_vec, team_equip_vec, team_status, team_num], dim=-1)
        team_out = self.team_slot_encoder(team_slot_in)  # [B, n_team, d_slot]

        shop_type_onehot = shop_block[..., self._s_type0 : self._s_type1]
        shop_pet_onehot = shop_block[..., self._s_pet0 : self._s_pet1]
        shop_food_onehot = shop_block[..., self._s_food0 : self._s_food1]
        shop_num = shop_block[..., self._s_num0 : self._s_num1]

        shop_pet_vec = self.pet_core(shop_pet_onehot)  # SAME module/weights as team_pet_vec above
        shop_food_vec = self.shop_food_proj(shop_food_onehot)
        shop_slot_in = th.cat([shop_type_onehot, shop_pet_vec, shop_food_vec, shop_num], dim=-1)
        shop_out = self.shop_slot_encoder(shop_slot_in)  # [B, n_shop, d_slot]

        opp_out = self.opp_encoder(opp_seg)
        global_in = th.cat([global_seg, summary_base_seg], dim=-1)
        global_out = self.global_encoder(global_in)

        if self.arch == "slot_attn":
            tokens = th.cat([team_out, shop_out], dim=1)  # [B, n_team+n_shop, d_slot], position order
            tokens = tokens + self.pos_embed
            attn_out, _weights = self.slot_attn(tokens, tokens, tokens)
            tokens = tokens + attn_out
            team_out = tokens[:, : self._n_team, :]
            shop_out = tokens[:, self._n_team :, :]

        team_flat = team_out.reshape(batch, self._n_team * self.d_slot)
        shop_flat = shop_out.reshape(batch, self._n_shop * self.d_slot)
        return th.cat([global_out, team_flat, shop_flat, opp_out], dim=-1)
