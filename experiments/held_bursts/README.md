# Why does the network hold its bursts?

**Question:** the replies network (eta 7.3) used fewer of its ready Elemental Bursts than beta 10.8. Why?

**What was done:** blind games against plain beta 10.8 on the locked test matchups, no search; at every decision
with a burst ready, the burst's probability, what was done instead, the round and the active character's HP; and for
every character that died with full energy, whether it was active (`explain_held_bursts.py`).

**Outcome:** beta 10.8 used 73% of its ready bursts, eta 7.3 53%. Almost every character that died with a full burst
was the active one (67 of 69 for beta 10.8, 69 of 76 for eta 7.3), so the burst was available: the network chose
something else, most often a fast tuning or card, and at any HP. The replies input did not fix this; it made it worse.
A likely cause noted at the time: skills carry no facts about their damage or effects in the input, only their
identity, so a burst's worth has to be learned from experience.
