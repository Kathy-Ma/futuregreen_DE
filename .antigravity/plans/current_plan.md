# Architect Plan: Consolidate Paper and Glass Datasets

## Overview

Create two consolidation scripts (`consolidate_paper.py` and `consolidate_glass.py`) to aggregate images from various raw datasets into a single processed directory for each class.

## File Structure

- **New Files**:
  - `processing_scripts/paper/consolidate_paper.py`
  - `processing_scripts/glass/consolidate_glass.py`
- **Output Directories**:
  - `data/processed/paper/`
  - `data/processed/glass/`

## Dependencies

- `os`
- `pathlib.Path`
- `PIL.Image`

## Step-by-Step Logic

### 1. Paper Consolidation (`consolidate_paper.py`)

**Source Directories**:

- `data/raw/TrashBox/paper/news paper`
- `data/raw/TrashBox/paper/paper`
- `data/raw/TrashBox/paper/paper cups`
- `data/raw/garbageClassification/paper`
- `data/raw/recyclableAndHouseholdWaste/newspaper`
- `data/raw/recyclableAndHouseholdWaste/office_paper`
- `data/raw/recyclableAndHouseholdWaste/paper_cups`
- `data/raw/trashNet/paper`

**Process**:

- Loop through each source directory.
- Use `rglob("*")` to find all files.
- Filter for valid image extensions (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`, `.tif`, `.tiff`).
- Open each valid image with `PIL.Image`, convert to `RGB`.
- Save to `data/processed/paper/paper_{N}.jpg` (with 95 quality).
- Maintain `global_counter`, `errors`, and `skipped` counts.
- Print a summary at the end.

### 2. Glass Consolidation (`consolidate_glass.py`)

**Source Directories**:

- `data/raw/TrashBox/glass`
- `data/raw/garbageClassification/brown-glass`
- `data/raw/garbageClassification/green-glass`
- `data/raw/garbageClassification/white-glass`
- `data/raw/recyclableAndHouseholdWaste/glass_beverage_bottles`
- `data/raw/recyclableAndHouseholdWaste/glass_cosmetic_containers`
- `data/raw/recyclableAndHouseholdWaste/glass_food_jars`
- `data/raw/trashNet/glass`

**Process**:

- Loop through each source directory.
- Use `rglob("*")` to find all files.
- Filter for valid image extensions (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`, `.tif`, `.tiff`).
- Open each valid image with `PIL.Image`, convert to `RGB`.
- Save to `data/processed/glass/glass_{N}.jpg` (with 95 quality).
- Maintain `global_counter`, `errors`, and `skipped` counts.
- Print a summary at the end.

## Edge Cases to Handle

- **Missing Source Directories**: The scripts should gracefully handle missing directories by printing a warning and continuing.
- **Corrupt Image Files**: `PIL.Image.open()` should be wrapped in a `try...except` block to catch and log any errors (e.g., UnidentifiedImageError), incrementing the `errors` count instead of crashing.
- **RGBA/Palette Images**: Ensure all images are safely converted to `"RGB"` via `.convert("RGBA").convert("RGB")` or similar safe logic, because some PNG/GIFs might have alpha channels or palettes that fail direct JPEG conversion.

## Handoff

This plan is ready for implementation by the Builder.
