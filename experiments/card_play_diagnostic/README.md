# Why does the bot tune its cards away?

**Question:** alpha 2.40 tuned away (converted to dice) many action cards that people would play. Is tuning them a
mistake (A), a fine choice (B), or does it depend on the card and situation (C)?

**What was done:** 3,130 real play-or-tune decisions were collected from benchmark games
(`collect_card_play_situations.py`), and each was played out both ways in paired counterfactual games, 23,040 in all
(`run_card_play_branches.py`), then summarized by card type, dice available and individual card
(`summarize_card_play_diagnostic.py`; report in `report.md`). `card_play_trend.py` follows card play over training.

**Outcome:** mostly C, with several card types closer to B: playing and tuning won about equally often for supports,
artifacts and talents; weapons were better played with few dice; a few individual cards (Gambler's Earrings, Raven Bow)
were clear mistakes to tune. There was no evidence that playing everything it tunes would make it stronger. Rewarding
card play directly (the delta runs, [docs/training.md](../../docs/training.md)) made it play many more cards without
winning more.
