# Reviews of the first training runs

Detailed reviews of alpha 1 and alpha 2, the first runs from scratch, written while deciding how to continue
(`reports/`):

- `alpha_1_statistics.md`: what alpha 1 learned over training, from 280,000 games: win rates, card play versus
  tuning, value calibration, seats and deck knowledge (`review_alpha_1_statistics.py`, with its sections in
  `alpha_1_statistics/`).
- `alpha_2_review.md`: alpha 2's checkpoint ladder, whether it plays reactions on purpose, how its card embeddings
  learned, and its results by deck origin (`review_*.py`).
- `external/chatgpt_review_of_alpha_1.pdf`: an outside analysis of alpha 1's exported games, made with ChatGPT from
  `scripts/training/export_for_external_analysis.py`'s bundle.

These shaped alpha 2 (smaller identity embeddings, the speed reward, a win-chance output) and the card-play work.
