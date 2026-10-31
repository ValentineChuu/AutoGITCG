# A critic that sees the opponent's hand

**Question:** in training, the value output (the critic) could be shown the opponent's hidden hand and dice, which it
can't see in a real game. Would that make it predict outcomes much better, and so make training better?

**What was done:** beta 10.8 self-play games with every kept decision encoded twice, with the normal view and with a
privileged view revealing the opponent's hand and dice (`privileged_view.py`); two copies of beta 10.8's value output
were fine-tuned on each and compared on held-out games (`compare_privileged_value.py`).

**Outcome:** revealing the opponent's current hand and dice barely improved the prediction: most of the uncertainty
is in future draws and rolls. The idea was dropped.
