Fusion module
=============

Purpose
-------
Provide strategies to fuse modality-specific features into a single
representation consumable by model adapters.

API
---
- `BaseFusion`: `fuse(features) -> fused`

Extend
------
Add new fusion strategies under this package that inherit from `BaseFusion`.
