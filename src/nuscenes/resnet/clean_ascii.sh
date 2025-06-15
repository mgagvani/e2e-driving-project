#!/usr/bin/env bash
# filepath: clean_ascii.sh
# Recursively clean all .py files under cwd, in-place.

set -euo pipefail

# require perl
command -v perl >/dev/null || { echo "perl not found – aborting." >&2; exit 1; }

find . -type f -name '*.py' -print0 \
| xargs -0 perl -i -pe '
  # map en/em-dash → hyphen
  s/[–—]/-/g;
  # map left/right single-quotes → apostrophe
  s/[\x{2018}\x{2019}]/\x27/g;
  # map left/right double-quotes → quote
  s/[\x{201C}\x{201D}]/\x22/g;
  # map ellipsis
  s/…/.../g;
  # non-breaking space → normal space
  s/\x{00A0}/ /g;
  # drop zero-width marks
  s/[\x{200C}\x{200D}\x{200E}\x{200F}]//g;
  # drop any other non-ASCII (keep tabs/newlines)
  s/[^[:ascii:]\t\n]//g;
'

echo "✅ All .py files cleaned."