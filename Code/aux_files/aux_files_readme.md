# Fixed inputs and settings

These files record the acquisition inventories, geometry, screening settings
and figure selection used by the November analysis. `figure3_sampling.h5`
contains selection and geometry only, with no heights or tide coefficients.

For the main workflow, copy `Code/input_paths.example.json` to the ignored
`Code/input_paths.json`. The numbered scripts create their runtime
`resources.json` under the work folder. `resources.example.json` is retained
for the lower-level batch interface.

See [DATA.md](../DATA.md) for the data list and [README.md](../README.md) for
the numbered scripts.

`figure3_reference_profiles.npz` also contains the fixed observed profiles used
in the paper comparison. `figure1_locator.json` and `paper_map_geometry.json`
contain drawing coordinates. See DATA.md for their sources and use.

`native_inventory191.json` matches the 191 Raster dates, cycles and passes.
`native_inventory190.json` is retained for the paper's original integer-phase
repair history. The two roles are described in DATA.md.
