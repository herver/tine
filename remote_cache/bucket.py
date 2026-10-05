# SPDX-FileCopyrightText: Amutable GmbH <https://amutable.com/>
# SPDX-License-Identifier: MPL-2.0

"""Basic read and write operations on the bucket.

Everyone reads, and reading is a plain HTTP `GET` of one key: no S3 API, no signing, no credentials on
a developer machine, and nothing to hand out. A builder writes, which needs an S3 key.
"""

import datetime
import hashlib
import hmac
import logging
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol, cast, override

log = logging.getLogger("bucket")

# Per socket operation, not per object: a large bundle on a slow line keeps passing this as long
# as bytes keep arriving, so it only has to be long enough for one round trip to a distant bucket.
TIMEOUT = 15

# How long an endpoint that could not be reached at all is left alone. A laptop behind a captive
# portal, or a firewall that drops rather than refuses, would otherwise wait out the timeout once
# per action, and a build has hundreds.
COOLDOWN = 60

# How big a pointer or a certificate can reasonably be. A pointer carries an ActionResult, which
# is ~100 bytes per output; a certificate is a few kilobytes. Bundles are read with a larger limit.
SMALL_OBJECT = 4 * 1024 * 1024

# Cloudflare's browser integrity check answers urllib's default agent with a 403; also useful for reading
# the bucket's logs.
USER_AGENT = "tine-cache-shim/0.1"


class Reader:
    """Reads a key over plain HTTP.

    A missing object is `None`; anything else raises `URLError`. A 403 or a 502 must not be recorded as
    "the bucket does not have this", or a broken endpoint quietly appears as an empty cache.

    An endpoint that cannot be reached at all is not asked again for `cooldown`.

    Nothing is read past `limit` bytes. The untrusted (in our model) bucket operator can serve anything
    under a key, and every check that would refuse an object runs after the download, so the download
    itself is the one cost that has to be bounded up front. An object over the limit is a `ValueError`.
    """

    def __init__(self, base_url: str, timeout: float = TIMEOUT, cooldown: float = COOLDOWN) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.cooldown = cooldown
        self._resting_until = 0.0
        self._lock = threading.Lock()

    def get(self, key: str, limit: int = SMALL_OBJECT) -> bytes | None:
        with self._lock:
            left = self._resting_until - time.monotonic()
        if left > 0:
            raise urllib.error.URLError(f"{self.base_url} was unreachable, not asked again for {left:.0f}s")
        request = self._request(key)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                declared = int(response.headers.get("Content-Length", 0))
                # Cast, because urlopen is typed loosely enough to hand back anything. One byte past
                # the limit is read so that a server that lies about the length is caught too.
                data = cast(bytes, response.read(limit + 1)) if declared <= limit else b""
                if declared > limit or len(data) > limit:
                    raise ValueError(f"{key} is over {limit} bytes, more than a reader takes")
                return data
        except urllib.error.HTTPError as error:
            # The error is itself an unread response, so it holds the connection until closed.
            error.close()
            if error.code == 404:
                return None
            raise
        except OSError as error:
            # Refused, timed out, reset, or unresolvable. urllib wraps some of these in URLError
            # and lets others through as they are; a caller gets one kind either way.
            with self._lock:
                self._resting_until = time.monotonic() + self.cooldown
            log.error(
                "%s is unreachable (%s), not asking again for %gs", self.base_url, error, self.cooldown
            )
            raise urllib.error.URLError(getattr(error, "reason", error)) from None
        finally:
            log.debug("GET %s", request.full_url)

    def _request(self, key: str) -> urllib.request.Request:
        return urllib.request.Request(f"{self.base_url}/{key}", headers={"User-Agent": USER_AGENT})

    def describe(self) -> str:
        return self.base_url


class Writer(Protocol):
    """Shape of `S3Writer`, test doubles implement it differently."""

    def put(self, key: str, data: bytes) -> None: ...

    def refresh(self, key: str) -> bool: ...

    def describe(self) -> str: ...


class S3Writer(Writer):
    """Writes a key with the S3 API.

    The only thing that needs a credential.

    Signed with hand-rolled SigV4: it's small and well documented, and avoids a ~120 MB python-boto3 SDK
    (we need nothing else from it). The bucket is part of the path. Keys are `<prefix>/<hex>`, under
    an optional `prefix` of plain names, so the canonical URI needs no escaping.
    """

    def __init__(
        self,
        endpoint: str,
        bucket: str,
        access_key: str,
        secret_key: str,
        region: str = "auto",
        secure: bool = True,
        prefix: str = "",
    ) -> None:
        self.endpoint = endpoint
        self.bucket = bucket
        self.access_key = access_key
        self.secret_key = secret_key
        self.region = region
        self.prefix = prefix
        self.base = f"{'https' if secure else 'http'}://{endpoint}/{bucket}"

    def _object(self, key: str) -> str:
        return f"{self.prefix}/{key}" if self.prefix else key

    def _signed(self, method: str, key: str, body: bytes, extra: dict[str, str]) -> urllib.request.Request:
        now = datetime.datetime.now(datetime.UTC)
        stamp, day = now.strftime("%Y%m%dT%H%M%SZ"), now.strftime("%Y%m%d")
        headers = {
            "host": self.endpoint,
            "x-amz-content-sha256": hashlib.sha256(body).hexdigest(),
            "x-amz-date": stamp,
            **extra,
        }
        signed = ";".join(sorted(headers))
        canonical = "\n".join(
            [
                method,
                f"/{self.bucket}/{self._object(key)}",
                "",
                *(f"{name}:{headers[name]}" for name in sorted(headers)),
            ]
        )
        canonical += f"\n\n{signed}\n{headers['x-amz-content-sha256']}"
        scope = f"{day}/{self.region}/s3/aws4_request"
        to_sign = f"AWS4-HMAC-SHA256\n{stamp}\n{scope}\n{hashlib.sha256(canonical.encode()).hexdigest()}"
        secret = f"AWS4{self.secret_key}".encode()
        for part in (day, self.region, "s3", "aws4_request"):
            secret = hmac.new(secret, part.encode(), hashlib.sha256).digest()
        signature = hmac.new(secret, to_sign.encode(), hashlib.sha256).hexdigest()
        headers["Authorization"] = (
            f"AWS4-HMAC-SHA256 Credential={self.access_key}/{scope}, "
            f"SignedHeaders={signed}, Signature={signature}"
        )
        del headers["host"]  # urllib sets it from the URL, and refuses to send it twice.
        return urllib.request.Request(
            f"{self.base}/{self._object(key)}",
            data=None if method == "GET" else body,
            method=method,
            headers=headers,
        )

    @override
    def put(self, key: str, data: bytes) -> None:
        with urllib.request.urlopen(self._signed("PUT", key, data, {}), timeout=TIMEOUT):
            pass
        log.info("PUT s3://%s/%s, %d bytes", self.bucket, self._object(key), len(data))

    @override
    def refresh(self, key: str) -> bool:
        """Give an existing object today's date.

        Return False if the object does not exist.
        """
        # A copy onto itself: one request, like the HEAD it replaces, and S3 makes a new object of
        # it with a new date. R2 does the same, checked: Last-Modified moves, the ETag does not,
        # an absent source is a 404 rather than an empty object, and the Content-Type is dropped,
        # which costs nothing because `put` never set one.
        copy = {
            "x-amz-copy-source": f"/{self.bucket}/{self._object(key)}",
            "x-amz-metadata-directive": "REPLACE",
        }
        try:
            with urllib.request.urlopen(self._signed("PUT", key, b"", copy), timeout=TIMEOUT):
                pass
        except urllib.error.HTTPError as error:
            error.close()
            if error.code == 404:
                return False
            raise
        return True

    @override
    def describe(self) -> str:
        if self.prefix:
            return f"s3://{self.bucket}/{self.prefix} at {self.endpoint}"
        return f"s3://{self.bucket} at {self.endpoint}"


class S3Reader(Reader):
    """Reads a key with a SigV4-signed GET, for a bucket that is not public.

    A missing key is a 404 to the key's owner, which keeps the miss/error distinction `Reader` rests on.
    """

    def __init__(self, writer: S3Writer, timeout: float = TIMEOUT, cooldown: float = COOLDOWN) -> None:
        super().__init__(writer.base + (f"/{writer.prefix}" if writer.prefix else ""), timeout, cooldown)
        self.writer = writer

    @override
    def _request(self, key: str) -> urllib.request.Request:
        request = self.writer._signed("GET", key, b"", {})
        # Unsigned, which SigV4 allows.
        request.add_header("User-Agent", USER_AGENT)
        return request


@dataclass(frozen=True)
class Bucket:
    reader: Reader
    writer: Writer | None = None

    def describe(self) -> str:
        if self.writer is None:
            return f"reading {self.reader.describe()}"
        return f"reading {self.reader.describe()}, writing {self.writer.describe()}"
