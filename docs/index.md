# On-the-Fly Data Assimilation

This package implements the on-the-fly method of data assimilation.

## Installation

At some point this package may be hosted on PyPI.
For now, one may install directly from GitHub.

I highly recommend the dependency manager `uv` (in place of `pip`).
Written in Rust, it's *much* faster than `pip`, and it's extremely convenient to use.
Following the `uv` [documentation](https://docs.astral.sh/uv/pip/packages/#installing-a-package), one may install from GitHub:

```bash
uv pip install "git+https://github.com/schilln/otf"
```

Optionally specify a specific version, commit, or branch:

```bash
# Install from a specific version:
uv pip install "git+https://github.com/schilln/otf@v0.1.0"

# Install from a specific commit:
uv pip install "git+https://github.com/schilln/otf@fb9289577c3e66f5f0503fc1c67cddb0e7eaf9e1"

# Install from a specific branch:
uv pip install "git+https://github.com/schilln/otf@main"
```

It seems one may replace `pip install` with `add` in any of the above commands to use `uv`'s dependency management and locking capabilities (highly recommended).

On the other hand, if one would truly rather not use `uv`, any of the above commands work with `uv` omitted.

## Usage

Two good ideas:

- Take a look at the [`examples`](https://github.com/schilln/otf/tree/main/examples) directory on GitHub.
  Not a bad idea to start with the Lorenz '63 system.
- Check out the API reference (see the link on the left of this page).
