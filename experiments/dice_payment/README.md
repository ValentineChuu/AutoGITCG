# Does automatic dice payment cost skills?

**Question:** the bot always pays with the engine's own legal dice selection. Does that ever spend dice it needed,
leaving its active character unable to use a skill another payment would have kept affordable?

**What was done:** 200 blind games of beta 10.8 against itself on the locked test matchups; at every card play and
tuning that paid dice, every other legal payment was checked (`count_dice_payment_losses.py`, `affordability.py`,
`skills_lost.py`).

**Outcome:** never. Of 4,604 payments, none cost a skill another payment would have kept, and the affordability rule
matched the engine on all 39,100 skill checks. Learning to pay dice was not worth adding.
