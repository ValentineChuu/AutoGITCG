# Training on rented machines

Most of the training ran on GPU machines rented by the hour from [vast.ai](https://vast.ai), alongside a desktop
(Ryzen 9 7900X, RTX 3070). The scripts that drove those machines were specific to our setup and are not part of this
repository; this page describes how it worked, so the approach can be repeated with any provider.

## What the hardware has to do

A combat training update has two halves:

- **game play** by the actor processes: the engine plays thousands of games while Python encodes every decision for
  the network. This is CPU work, one core per actor process, so single-core speed matters.
- **the learner's PPO update**: GPU arithmetic over the collected steps.

With the original Node engine, game play dominated: the best machines had two GPUs (actors on one, learner on the
other, `--actor-device`) and fast desktop-class cores (Ryzen 9, Core i9, Threadripper) rather than many slow server
cores (EPYC, Xeon). After the C++ engine (`engine/`) made game play 20-40 times faster, the learner became the
limit, and one fast GPU was what mattered: the final PvP run (theta 1) took about 1.1 minutes per update on a single
RTX 4090 with a 64-core EPYC, about 11 hours for 600 updates.

Deck-builder training is different: it is almost all game play, so many cores matter more than the GPU.

Other requirements: on-demand (not interruptible) instances, a recent CUDA driver, about 50 GB of disk.

## Workflow

- **One run folder per training run** (`runs/<name>/`), the same on every machine. Training resumes from
  `checkpoint_latest.pt` with the same command, so a run can move between machines: stop it right after a
  checkpoint, copy the run folder, start the same command elsewhere. Updates after the last checkpoint are replayed,
  so there are no gaps or duplicates; `machines.jsonl` in the run folder records which machine ran which updates.
- **Code goes up, results come down**, both with `rsync`: the code, the game data and the models up; the run folder
  down every few minutes while the run is going. The C++ engine is built as a static binary, so the rented machine
  needs no compiler.
- **Crashes restart from the latest checkpoint** automatically (a few times at most), with each exit code logged;
  a deliberate stop (SIGTERM) is not restarted.
- **A guard per machine**, running on the desktop: it pulls the results regularly, warns when the account credit is
  low, and when no training or test job has run for 30 minutes it pulls one last time and destroys the machine,
  only if that last pull succeeded. Rented machines were never merely stopped: restarting a stopped instance can wait
  a long time for its GPU, with the data stuck on it meanwhile.
- **Checkpoint tests run elsewhere**: every 50 updates, a separate machine (the desktop or a small home server)
  played each new checkpoint against the reference network, so tests never slowed training. A run stopped itself
  after two tests in a row without a new best.

## Lessons

- **Measure one job before renting a fleet.** A batch of tests spread over several cheap machines cost twice the
  estimate, because each job was limited by its single Python thread and its memory, not by the cores we paid for.
- **Check a new machine before trusting it**: run the test suite and look at the load. One host was so oversubscribed
  that the tests took 20 times longer than elsewhere; a consumer host without ECC memory crashed three times in 40
  minutes with segfaults in unrelated places.
- **Keep files written on one side out of folders the other side syncs**, or the next sync overwrites them.
