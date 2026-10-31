# Students: copying the search into the network

**Question:** the look-ahead and the deep judge play much better than the network alone, but they are slow. Can a
network learn to play like them in one forward pass, by training on their choices?

**What was done:** the search played many games and recorded its choice at each searched decision (the labels,
`record_teacher_labels.py`, `label_*.py`, `teacher_label.py`); students were trained on them from beta 10.8 or
gamma 5 (`train_student.py`, `student_*.py`) with many label rules: copy the played move, copy the best-scoring move,
only clear overrides, soft targets shifted by the search's score gaps, and more. These are the epsilon (look-ahead
labels) and zeta (deep-judge labels) networks in [docs/training.md](../../docs/training.md); a move-value output was
also tried (`compare_move_values.py`).

**Outcome:** no student beat the network the labels came from: the best were at 48-53% against beta 10.8, and a rule
that kept only clear wins fell to 22%. Most labels are close calls: the search's best move is barely better than the
network's own, within the noise of a few deals, so the students mostly learned noise, and every label saying "not the
network's move" pulled them away from good play everywhere.

`notes.md` is the original plan and log.
