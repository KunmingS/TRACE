"""Read single members out of a remote ZIP over HTTP range requests.

Lets `vtrace demo` take its videos from the dataset's own archive without
downloading the archive. A ZIP keeps its index (the central directory) at the
end of the file, so one range request for the tail is enough to learn where
every member lives, and a second range request per member fetches just that
member's bytes.

Stdlib only: `zipfile.ZipFile` accepts any seekable binary file object, and
`HttpFile` below is one backed by ranged GETs.
"""
from __future__ import annotations

import io
import time
import urllib.error
import urllib.request
import zipfile

# Some hosts (CaltechDATA's S3 front end among them) reject the default
# `Python-urllib/x.y` agent with a 403. Identify the tool instead.
USER_AGENT = "v-trace (+https://github.com/KunmingS/TRACE)"

_TIMEOUT = 120

# A single video can be most of a gigabyte, which is long enough for a transient
# network error to be likely rather than exceptional. Retry the range request
# rather than losing the whole file to one dropped connection.
_ATTEMPTS = 4
_BACKOFF = 2.0


class HttpFile(io.RawIOBase):
    """A read-only, seekable file over an HTTP resource that supports ranges."""

    def __init__(self, url: str, size: int):
        self.url = url
        self.size = size
        self.pos = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        else:
            self.pos = self.size + offset
        self.pos = max(0, min(self.size, self.pos))
        return self.pos

    def read(self, n: int = -1) -> bytes:
        if n < 0 or self.pos + n > self.size:
            n = self.size - self.pos
        if n <= 0:
            return b""
        request = urllib.request.Request(
            self.url,
            headers={
                "Range": f"bytes={self.pos}-{self.pos + n - 1}",
                "User-Agent": USER_AGENT,
            },
        )
        data = self._get(request)
        self.pos += len(data)
        return data

    def _get(self, request) -> bytes:
        last = None
        for attempt in range(_ATTEMPTS):
            try:
                with urllib.request.urlopen(request, timeout=_TIMEOUT) as response:
                    return response.read()
            except urllib.error.HTTPError as exc:
                # 4xx other than 408/429 will not become true by waiting.
                if exc.code not in (408, 429) and exc.code < 500:
                    raise
                last = exc
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
                last = exc
            if attempt < _ATTEMPTS - 1:
                time.sleep(_BACKOFF * (2 ** attempt))
        raise last


def open_remote(url: str, size: int) -> zipfile.ZipFile:
    """Open the ZIP at `url` (of `size` bytes) for member-wise reading."""
    return zipfile.ZipFile(HttpFile(url, size))
