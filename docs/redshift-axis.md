# The redshift axis

The compilation runs from z = 0 (peculiar velocities, most below z = 0.08) to
z = 1.5 (clustering). No single scale serves both, so four are available and
each figure picks one:

- `linear` — right for a PV-only or RSD-only panel; useless for a mixed one,
  where every PV point stacks into the leftmost 5%.
- `symlog` — linear below z = 0.1, log above. **The default for mixed panels.**
- `log1p` — log(1+z). Also keeps z = 0, but 1+z only runs 1 → 1.1 across the PV
  range, so it barely expands the low-z end.
- `log` — silently drops the six z = 0 rows. Never on a panel with PV data.

Section 4 of the notebook plots the same data on all four.

Points at nearly coincident redshifts are dodged horizontally. Two guards on
that, because a dodge is a lie told for legibility and must not become a lie
about the data: `floor=0` stops the six rows quoted at exactly z_eff = 0 from
being spread to *negative* redshift, and `min_gap` enforces a separation
between all neighbours rather than only within a tied block — without it two
adjacent blocks can land a thousandth apart, invisible for markers and fatal
for per-point labels. Every figure that dodges says so in its caption.
