# The trained networks

Every network from the training runs is published on [the releases page](https://github.com/ValentineChuu/AutoGITCG/releases), not in the repository:
162 files, about 1.1 GB, each a frozen network without optimizer state. Each run is its own release (`theta-1`,
`alpha-2`, ... `chudledee-v3`, and `epsilon` and `zeta` for the students), with one zip holding the run's network every
50 updates and a description of what the run tried. Unzip into `models/`, where the scripts look for them.
[model_manifest.json](model_manifest.json) lists every file with its family, run, update, the run's idea and a SHA-256.

To try the bots, `theta-1.zip` and `beta-10.zip` are enough. The tests also need `alpha-2.zip`, `eta-7.zip` and
`chudledee-v3.zip`. With the [GitHub CLI](https://cli.github.com):

```sh
for run in theta-1 beta-10 alpha-2 eta-7 chudledee-v3; do
    gh release download $run --repo ValentineChuu/AutoGITCG --dir models
done
cd models && for archive in *.zip; do unzip -o "$archive" && rm "$archive"; done
```

Every run at once:

```sh
gh release list --repo ValentineChuu/AutoGITCG --limit 100 --json tagName --jq '.[].tagName' |
    while read -r run; do gh release download "$run" --repo ValentineChuu/AutoGITCG --dir models; done
cd models && for archive in *.zip; do unzip -o "$archive" && rm "$archive"; done
```

## Names

A combat network is named `<family> <run>.<update / 50>`: one file per 50 training updates, so theta 1 at update 650
is `theta-1.13.pt`. A network between milestones adds the updates past the last one: eta 6 stopped at update 30, so
its network is `eta-6.0.30.pt` (and theta 1 at update 670 would be theta 1.13.20). Students are published at their
final epoch (`epsilon-2.pt`). The families and what each run tried are in [training.md](training.md).

Chudledee, the deck builder, is numbered the same way within its three versions: `chudledee-v1.1.pt` is v1 at update
50, and each version's last file is its final network: v1.6, v2.4 and v3.4 ([deck-builder.md](deck-builder.md)).

## The ones to start with

| File | What it is |
|---|---|
| `theta-1.13.pt` | the PvP bot: best on current tournament decks (69.4% vs beta 10.8; [results.md](results.md)) |
| `beta-10.8.pt` | the NPC bot, and the reference every later network was measured against |
| `alpha-2.40.pt` | the first strong network, trained from scratch |
| `eta-7.3.pt` | the best network with the replies input before the PvP retrain |
| `chudledee-v3.4.pt` | the deck builder (Chudledee v3, final) |

Theta and eta read the previews and replies inputs, so games they play need the engine to compute them; the scripts
check each network's config and turn this on. For example, theta 1.13 with the quick judge against plain beta 10.8 on
the meta test:

```sh
python scripts/evaluation/judge_duel.py --first models/theta-1.13.pt --first-judge quick \
    --second models/beta-10.8.pt --second-judge none --benchmark meta --matchups 20 --output runs/duel/example
```

## Not released

- Runs whose folders weren't kept: alpha 0 and 3, gamma 1-3, delta 9.
- The beta 3-9 diagnosis runs: 20-60-update tests of training settings, not networks to play.
- Eta 2: it uses an older shape of the board-prediction output, which the current code can't load.
- Eta 8: stopped before it saved a trained network; it only has its starting point, eta 7.3.
- Students whose labels can't be matched to a documented experiment.

The files are built with `scripts/training/export_model_release.py` from the run folders listed in
`scripts/training/model_release_plan.json`; every file is loaded back into its network before it is listed.
