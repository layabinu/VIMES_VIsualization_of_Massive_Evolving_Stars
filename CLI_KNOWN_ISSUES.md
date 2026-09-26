# CLI Known Issues

Tracked issues and limitations for the VIMES command-line interface
(`vimes` and `vimes-preprocess`). Last reviewed after the `argparse` →
`typer` migration.

## Open

### 1. Default paths resolve into the package directory

`vimes` (`animate.py`) defaults `frames` to `BASE_DIR / "frames_data.npz"`,
and `vimes-preprocess` (`preprocess.py`) defaults `hdf5`/`out` to
`Path(__file__).parent / ...`. These resolve into `src/vimes/` — the
installed package directory — rather than the user's working directory or a
data directory.

For a data-processing tool this is surprising: running the commands from
anywhere looks for data inside the site-packages copy of the package.

Suggested fix: base defaults on `Path.cwd()` (or an explicit data dir /
option).

### 2. `temp_to_color.py` has its own stale entry point

`temp_to_color.py` ends with:

```python
if __name__ == "__main__":
    add_temperatures_and_rgb()
```

This is a third, inconsistent entry point that calls
`add_temperatures_and_rgb()` with the module-level constants
(`HDF5_PATH`, `FRAMES_NPZ`, `OUTPUT_NPZ`) instead of CLI arguments. It is
not wired into `pyproject.toml`.

Suggested fix: remove the block, or route it through `cli_app`.

### 3. No `python -m vimes` support

There is no `src/vimes/__main__.py`, so only the console scripts
(`vimes`, `vimes-preprocess`) are invocable. `python -m vimes` fails.

Optional: add `__main__.py` if module invocation is desired.

### 4. CLI apps live inside the worker modules

`cli_app`/`animate` are defined in `animate.py` and
`cli_app`/`preprocess` in `preprocess.py`, coupling the CLI to the
visualization/preprocessing code. This is acceptable, but consolidating both
into a single `cli.py` would give one place for enums, validation helpers,
and typer apps.

## Resolved (during the argparse → typer migration)

- **`out` was existence-validated.** `preprocess.py` applied the input-file
  existence check to the *output* frames path, so a normal first run failed
  until the file already existed. Fixed: `out` is now a plain `Path`; only
  `hdf5` (and `frames` in `animate`) are guarded.
- **Duplicated path validator.** The same existence-check closure was
  copy-pasted with inconsistent names/messages. Removed in favor of a single
  `typer.Exit` guard per command.
- **Enum help text showed `{ScalingType.LOG,ScalingType.LINEAR}`.** This was
  an argparse `choices`/`__str__` workaround. typer renders enum options
  natively (`<log|linear>`); the workaround and dead `__str__` methods were
  dropped.
- **Positional arguments ignored their `default`.** argparse requires
  `nargs="?"` for a positional default to take effect. typer's
  `typer.Argument(...) = default` produces optional positionals natively.
- **Stale error message** `"Run compas_preprocess.py first."` → now
  `"Run vimes-preprocess first."`.
