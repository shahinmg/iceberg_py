#!/usr/bin/env python
"""Quick test to verify iceberg_model installation."""

print("Testing iceberg_model installation...")
print("=" * 60)

try:
    from iceberg_model import Iceberg
    print("✓ Iceberg class imported")
except ImportError as e:
    print(f"✗ Failed to import Iceberg: {e}")
    exit(1)

try:
    from iceberg_model import constants
    print(f"✓ Constants imported (factor={constants.DEFAULT_TRANSFER_COEFFICIENT_FACTOR})")
except ImportError as e:
    print(f"✗ Failed to import constants: {e}")
    exit(1)

try:
    from iceberg_model.simulation import IcebergMeltSimulation
    print("✓ IcebergMeltSimulation imported")
except ImportError as e:
    print(f"✗ Failed to import IcebergMeltSimulation: {e}")
    exit(1)

try:
    berg = Iceberg(length=200, dz=10)
    print(f"✓ Created iceberg: {berg}")
except Exception as e:
    print(f"✗ Failed to create iceberg: {e}")
    exit(1)

try:
    keel = berg.keeldepth()
    print(f"✓ Calculated keel depth: {keel:.1f}m (type: {type(keel).__name__})")
except Exception as e:
    print(f"✗ Failed to calculate keel: {e}")
    exit(1)

try:
    geom = berg.init_iceberg_size()
    print(f"✓ Initialized geometry: volume={geom.totalV.values:.2e} m³")
except Exception as e:
    print(f"✗ Failed to initialize geometry: {e}")
    exit(1)

print("=" * 60)
print("✓ ALL TESTS PASSED! Package installed correctly.")