# Third-party notices

The original project code and pretrained model weights are provided under the
[MIT License](LICENSE). The following notices preserve the terms and attribution
of included upstream material.

## Stable Baselines

`python/sap_ppo/train/kl_ppo.py` contains code adapted from `MaskablePPO.train`
in [sb3-contrib](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib),
which builds on [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3).
Their license and attribution texts are preserved without modification:

- [sb3-contrib LICENSE](third_party/licenses/sb3-contrib/LICENSE)
- [Stable-Baselines3 LICENSE](third_party/licenses/stable-baselines3/LICENSE)
- [Stable-Baselines3 NOTICE](third_party/licenses/stable-baselines3/NOTICE)

## sapai

The canonical pet and food IDs in `data/turtle_catalog_v1.json` and the example
snapshot in `fixtures/parity_cases/sample_case.json` derive from
[sapai](https://github.com/manny405/sapai). Its repository license and the
license notice embedded in its game-data module are preserved:

- [sapai LICENSE](third_party/licenses/sapai/LICENSE)
- [sapai game-data license](third_party/licenses/sapai/DATA_LICENSE)

## External game materials

[SAP-Calculator](https://github.com/robertley/SAP-Calculator) and the
[sap-replay-bot renderer](https://github.com/RuihanCao/sap-replay-bot/tree/ca06ba34bf647b7876d15922c9b24ab0d1dd16a3)
are downloaded separately. Replay-bot's package declares the ISC License;
its source is used without modification. Game artwork is read from these local checkouts;
no original game artwork is included in this repository.

This project's MIT License does not grant rights to Super Auto Pets, its game
artwork, third-party gameplay records, or separately downloaded dependencies.
