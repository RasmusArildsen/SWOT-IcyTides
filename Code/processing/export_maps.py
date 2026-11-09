"""Copy the fitted DEM and amplitude maps into plainly named HDF5 files."""

from pathlib import Path
import argparse
import h5py


def copy_product(source, target, fields):
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp.h5")
    with h5py.File(source) as inputs, h5py.File(temporary, "w") as output:
        assert inputs.attrs["complete"] == 1
        for name, exported_name in fields.items():
            inputs.copy(name, output, name=exported_name)
        output.attrs.update(dict(inputs.attrs))
        output.attrs["source"] = str(source.resolve())
        output.attrs["complete"] = 1
    temporary.replace(target)
    print(target, flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--products", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    copy_product(
        args.products / "figure1.h5",
        args.output_dir / "dem_100m.h5",
        {"x": "x", "y": "y", "mean": "elevation"},
    )
    fields = {"x": "x", "y": "y"}
    for constituent in ("m2", "s2", "k1", "o1"):
        for suffix in ("amp", "cats_amp"):
            name = constituent + "_" + suffix
            fields[name] = name
    copy_product(
        args.products / "maps.h5", args.output_dir / "tidal_amplitudes_250m.h5", fields
    )


if __name__ == "__main__":
    main()
