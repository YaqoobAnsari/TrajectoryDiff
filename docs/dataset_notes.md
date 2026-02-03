# RadioMapSeer Dataset Documentation

## Overview

**RadioMapSeer** is a pathloss radio map dataset for urban outdoor environments, generated using WinProp ray-tracing software.

**Key Insight**: While the original project plan assumed indoor environments, RadioMapSeer is outdoor/urban. However, **our trajectory-based approach still applies**:
- Streets/sidewalks = walkable paths (like corridors)
- Buildings = obstacles (like walls)
- Private areas = blind spots (no crowdsourced data)

## Dataset Specifications

| Property | Value |
|----------|-------|
| **Format** | PNG images (8-bit grayscale) + JSON metadata |
| **Resolution** | 256×256 pixels |
| **Spatial Resolution** | 1 meter per pixel |
| **Coverage** | 256m × 256m per map |
| **City Maps** | 701 unique maps |
| **Transmitters per Map** | 80 |
| **Total Samples** | 56,080 per propagation model |
| **Cities** | Ankara, Berlin, Glasgow, Ljubljana, London, Tel Aviv |

## Pathloss Value Encoding

| Property | Value |
|----------|-------|
| **Min Pathloss** | -186 dB |
| **Max Pathloss** | -47 dB |
| **Threshold** | -127 dB |
| **PNG Encoding** | Linear mapping to 0-255 grayscale |

**Conversion formula**:
```python
# PNG to dB
pathloss_db = (png_value / 255) * 139 + (-186)  # Approximate

# dB to PNG
png_value = int((pathloss_db - (-186)) / 139 * 255)
```

## Dataset Variants

| Variant | Description | Recommended |
|---------|-------------|-------------|
| **IRT2** | Intelligent Ray Tracing, 2 max interactions | Yes (simpler) |
| **IRT4** | Intelligent Ray Tracing, 4 max interactions | More realistic |
| **DPM** | Dominant Path Model | Fastest but less accurate |
| **3D IRT2** | 3D variant | For 3D research |
| **IRT2HighRes** | Higher resolution variant | **Recommended** |

## Download Links

- **Primary**: [IEEE DataPort](https://ieee-dataport.org/documents/dataset-pathloss-and-toa-radio-maps-localization-application)
- **Alternative**: [RadioMapSeer GitHub Page](https://radiomapseer.github.io/) (Google Drive)
- **Recommended file**: `IRT2HighRes.zip` (~930 MB)

## Data Organization (Expected)

```
RadioMapSeer/
├── city_maps/           # Building footprints (256×256 PNG)
│   ├── ankara/
│   ├── berlin/
│   └── ...
├── radio_maps/          # Pathloss maps (256×256 PNG)
│   ├── IRT2/
│   │   ├── map_001_tx_01.png
│   │   └── ...
│   └── IRT4/
├── tx_locations/        # Transmitter positions (JSON or PNG)
└── metadata/            # Simulation parameters
```

*Note: Exact structure to be verified after download.*

## Trajectory Sampling Adaptation

For outdoor urban scenarios, trajectory sampling represents:

1. **Pedestrian paths**: Walking along sidewalks, crossing streets
2. **Vehicle trajectories**: Driving along roads
3. **Delivery routes**: Specific areas of interest

**Key differences from indoor**:
- Larger scale (city blocks vs. building floors)
- More open areas (streets vs. corridors)
- Different obstacle types (buildings vs. walls)

## Citation

```bibtex
@misc{yapar2022dataset,
  title={Dataset of Pathloss and ToA Radio Maps With Localization Application},
  author={Yapar, Cagkan and Levie, Ron and Kutyniok, Gitta and Caire, Giuseppe},
  year={2022},
  doi={10.21227/0gtx-6v30},
  publisher={IEEE Dataport}
}
```

## License

Creative Commons Attribution 4.0 (CC BY 4.0)

---

*Last updated: 2026-02-03*
