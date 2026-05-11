# Data

Place OCEL 1.0 JSON log files here before running experiments.

## Included in repository

| File | Description | Size |
|------|-------------|------|
| `example_log.jsonocel` | Order management process, 3 object types, 23 events, 15 objects | ~5 KB |

## Download automatically

Run once to download all other logs needed for Experiment 3:

```bash
python scripts/download_data.py
```

This downloads:

| File | Description | Source |
|------|-------------|--------|
| `running_example.jsonocel` | Package fulfilment, 2 object types | Liss et al. 2023 |
| `p2p_normal.jsonocel` | Procure-to-Pay process, 5 object types | ocpa library |

## Using your own log

Any OCEL 1.0 JSON file can be used:

```bash
python run_conformance.py data/your_log.jsonocel
```

The OC-PN will be discovered automatically via pm4py.

## Format

All logs must be OCEL 1.0 JSON format with the following structure:

```json
{
  "ocel:global-log": { "ocel:object-types": ["order", "item"] },
  "ocel:events": {
    "e1": {
      "ocel:activity":  "Place Order",
      "ocel:timestamp": "2024-01-01T00:00:01+00:00",
      "ocel:omap":      ["o1", "i1"],
      "ocel:vmap":      {}
    }
  },
  "ocel:objects": {
    "o1": { "ocel:type": "order",  "ocel:ovmap": {} },
    "i1": { "ocel:type": "item",   "ocel:ovmap": {} }
  }
}
```

## Note on version control

Only `example_log.jsonocel` is committed to the repository (5 KB).
The other logs are excluded via `.gitignore` and must be downloaded
with `python scripts/download_data.py` before running Experiments 3–4.