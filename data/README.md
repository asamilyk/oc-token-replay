# Data

Place OCEL 2.0 JSON files here.

## OCEL 2.0 Order Management Log (used in Experiment 2)

Download from: https://ocel-standard.org/

File: `order-management.jsonocel`

Then run:
```bash
python experiments/run_experiments.py --exp2 --ocel data/order-management.jsonocel
```

## Expected OCEL 2.0 JSON format

```json
{
  "ocel:events": {
    "e1": {
      "ocel:activity": "Place Order",
      "ocel:timestamp": "2023-01-01T10:00:00Z",
      "ocel:omap": ["o1", "i1", "i2"]
    }
  },
  "ocel:objects": {
    "o1": { "ocel:type": "order" },
    "i1": { "ocel:type": "item" },
    "i2": { "ocel:type": "item" }
  }
}
```
