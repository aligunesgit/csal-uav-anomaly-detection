# Dataset

The dataset is **not included** in this repository due to file size.

## Download

Download the **Galician Rivers Multispectral Anomaly Detection Dataset** from Zenodo:

> J. López-Fandiño, À. Ordóñez, P. Quesada-Barriuso, A. S. Garea, F. Argüello, and D. B. Heras,
> "Galician rivers multispectral anomaly detection dataset," 2025.
> https://doi.org/10.5281/zenodo.14852117

## Expected directory structure

After downloading, place the scene folders directly inside this `data/` directory:

```
data/
├── z1/
│   ├── z1.raw
│   ├── z1_gt.pgm
│   └── z1_seg.raw
├── z2/
│   ├── z2.raw
│   ├── z2_gt.pgm
│   └── z2_seg.raw
├── e1/
│   ├── e1.raw
│   ├── e1_gt.pgm
│   └── e1_seg.raw
└── e2/
    ├── e2.raw
    ├── e2_gt.pgm
    └── e2_seg.raw
```

## Scene summary

| Scene | Width (px) | Height (px) | Anomaly % |
|-------|-----------|------------|-----------|
| Z1    | 3807      | 2141       | 3.95      |
| Z2    | 2081      | 957        | 12.51     |
| E1    | 3629      | 961        | 4.72      |
| E2    | 1094      | 707        | 3.30      |

Images were acquired with a MicaSense RedEdge sensor at 8.2 cm/pixel (120 m altitude) over riparian zones in Galicia, Spain (summer 2018). Five spectral bands: Blue (475 nm), Green (560 nm), Red (668 nm), RedEdge (717 nm), NIR (840 nm).
