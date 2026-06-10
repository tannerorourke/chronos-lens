import hashlib
import concurrent.futures
import io
from collections.abc import Callable
from concurrent.futures import Future
from pathlib import Path

import logging
logger = logging.getLogger(__name__)

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.exceptions import BotoCoreError, ClientError

from src.utils.system import AWS_S3_BUCKET, AWS_REGION

class S3Client:
    """ 
    Manages upload, download, rename, validation, and sync between a local
    directory and s3://<bucket>/<s3_subdir>/<local_dir.name>
 
    Use as a context manager to ensure executor cleanup:
        with S3Client(local_dir) as client:
            client.upload("model.pt")
    """
    _MULTIPART_CHUNK: int = 8 * 1024 * 1024
    _TRANSFER_CONFIG: TransferConfig = TransferConfig(
        multipart_threshold=_MULTIPART_CHUNK,
        multipart_chunksize=_MULTIPART_CHUNK,
    )
    
    def __init__(
        self, 
        local_dir: Path, 
        bucket: str = AWS_S3_BUCKET, 
        region: str = AWS_REGION,
        s3_subdir: str = "",
        strict: bool = True
    ):
        self._client = boto3.client("s3", region_name=region)
        self._sts = boto3.client("sts", region_name=region)
        self._bucket = bucket
        self._region = region
        self._executor = concurrent.futures.ThreadPoolExecutor()
        self.local_dir = Path(local_dir)
        self._prefix = f"{s3_subdir}/{local_dir.name}" if s3_subdir else local_dir.name
        self.strict = strict
        
        if strict:
            if not self._is_live(): 
                raise RuntimeError("AWS credentials not available.")
            if not self.local_dir.exists():
                raise FileNotFoundError(f"../{self.local_dir.parts[-2:]} does not exist.")
    
    
    # --- core wrapper
    def _run(self, context: str, fn: Callable[[], None]) -> bool:
        """
        Execute a void S3 operation synchronously and return True on success.
        In non-strict mode, catches expected exceptions, logs, and returns False.
        """
        try:
            fn()
            return True
        except (ClientError, BotoCoreError, OSError, ValueError) as e:
            if self.strict:
                raise
            logger.warning("[S3Client] %s failed: %s", context, e)
            return False
    
    def _call(self, 
        context: str, 
        _fn: Callable[[], None], 
        _async: bool = False
    ) -> bool | Future[bool]:
        """ Dispatch a void S3 operation via _run, optionally onto the thread pool. """
        if _async:
            return self._executor.submit(self._run, context, _fn)
        return self._run(context, _fn)
    
    
    # --- path helpers
    def _local(self, filename: str) -> Path:
        return self.local_dir / filename

    def _key(self, filename: str) -> str:
        return f"{self._prefix}/{filename}"
    
    def _as_posix(self, file: str | Path) -> str:
        """ 
        Resolve `file` to POSIX-style relative string for use in _local/_key. 
        """
        p = Path(file)
        if p.is_absolute():
            return p.relative_to(self.local_dir).as_posix()
        return p.as_posix()
    
    
    # --- checksum validation    
    def _validate_checksum(self, file: str) -> None:
        """ 
        Compare the local file's MD5 against the S3 ETag. Handles both 
        single-part and multipart ETags, raising ValueError on mismatch.
 
        Note: multipart ETag computation assumes uploads used _TRANSFER_CONFIG.
        If the object was uploaded externally with a different chunk size, the
        multipart ETag will not match. 
        """
        local_path = self._local(file)
        etag = self._client.head_object(
            Bucket=self._bucket, 
            Key=self._key(file)
        )["ETag"].strip('"')
        local_etag = (self._multipart_md5(local_path) if "-" in etag else
                      self._md5(local_path))
        
        if local_etag != etag:
            raise ValueError(f"Checksum mismatch for {file!r}: local={local_etag} s3={etag}")
        
            
    @staticmethod
    def _md5(path: Path) -> str:
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
    
    
    def _multipart_md5(self, path: Path) -> str:
        """
        Replicates S3's multipart ETag: MD5 of the concatenated per-part MD5
        digests (raw bytes), suffixed with the part count.
        e.g. "d8e8fca2dc0f896fd7cb4cb0031ba249-3"
        """
        part_digests: list[bytes] = []
        with open(path, "rb") as f:
            while chunk := f.read(self._MULTIPART_CHUNK):
                part_digests.append(hashlib.md5(chunk).digest())
        return f"{hashlib.md5(b"".join(part_digests)).hexdigest()}-{len(part_digests)}"
        
        
    # --- public methods
    def upload(
        self, 
        file: str | Path,
        _async: bool = False,
        _overwrite: bool = False,
        _validate: bool = False
    ) -> bool | Future[bool]:
        """
        Upload `<local_dir>/<filename>` to `S3://<bucket>/runs/<run-id>/<filename>`.
        If _validate=True, checksum validation runs after the upload completes.
        If If _validate=True and _async=True, both steps run on the executor thread
        and the Future resolves only after validation.
        """
        file = self._as_posix(file)
        def _do() -> None:
            logger.info(f"[S3Client.upload] uploading '{file!r}' to S3")
            if not _overwrite and self.exists(file):
                raise FileExistsError(f"'{file!r}' already exists in S3. Use _overwrite=True to overwrite.")
            self._client.upload_file(
                str(self._local(file)),
                self._bucket,
                self._key(file),
                Config=self._TRANSFER_CONFIG,
            )
            if _validate:
                self._validate_checksum(file)
                
        return self._call(f"upload({file})", _do, _async=_async) 
    
    def fetch(
        self, 
        file: str | Path,
        _overwrite: bool = False,
        _async: bool = False,
        _validate: bool = False
    ) -> bool | Future[bool]:
        """ 
        Fetch s3://<bucket>/<s3_subdir>/<local_dir.name> to disk at `<local_dir>/<filename>`.
        """
        file = self._as_posix(file)
        def _do():
            self._local(file).parent.mkdir(parents=True, exist_ok=True)
            logger.debug(f"[S3Client.fetch] fetching '{file!r}' to local disk")
            if not _overwrite and self._local(file).exists():
                raise FileExistsError(f"'{file!r}' already exists locally. Use _overwrite=True to overwrite.")
            self._client.download_file(
                self._bucket, 
                self._key(file), 
                str(self._local(file))
            )
            if _validate:
                self._validate_checksum(file)
        return self._call(f"fetch({file})", _do, _async=_async)
      
    def upload_folder(
        self,
        folder: str | Path,
        _async: bool = False,
        _overwrite: bool = False,
        _validate: bool = False
    ) -> bool | Future[bool]:   
        folder = self._as_posix(folder)
        ldir = self._local(folder)
        assert ldir.is_dir(), f"Folder {folder} not found locally"
        
        def _do():
            for file in ldir.iterdir():
                fn = ldir / file
                self.upload(file=fn, _async=False, _overwrite=_overwrite, _validate=_validate)
        return self._call(f"upload_folder({folder})", _do, _async=_async)
    
      
    def fetch_folder(
        self, 
        folder: str | Path,
        _overwrite: bool = False,
        _async: bool = False,
        _validate: bool = False
    ) -> bool | Future[bool]:
        folder = Path(folder).as_posix().strip("/")
        s3_folder_prefix = f"{self._prefix}/{folder}/"
        
        def _do() -> None:
            paginator = self._client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=self._bucket, Prefix=s3_folder_prefix):
                for obj in page.get("Contents", []):
                    s3_key = obj["Key"]
                    if s3_key.endswith("/"):
                        continue  # skip folder placeholder objects
                    # key = "{self._prefix}/{relative}" — strip the prefix to get
                    # the relative path, then reuse _local/_key as normal.
                    relative = s3_key[len(f"{self._prefix}/"):]
                    logger.debug("[S3Client.fetch_folder] fetching '%s'", relative)
                    if not _overwrite and self._local(relative).exists():
                        logger.warning("[S3Client.fetch_folder] '%s' exists locally, skipping", relative)
                        continue
                    self._local(relative).parent.mkdir(parents=True, exist_ok=True)
                    self._client.download_file(self._bucket, s3_key, str(self._local(relative)))
                    if _validate:
                        self._validate_checksum(relative)
        return self._call(f"fetch_folder({folder})", _do, _async=_async)
        
    def rename(
        self, 
        old_file: str | Path, 
        new_file: str | Path, 
        _async: bool = False, 
        _validate: bool = False
    ) -> bool | Future[bool]:
        """
        Rename an S3 key `s3://<bucket>/<s3_subdir>/<local_dir.name>/<old_file>` to
        `s3://<bucket>/<s3_subdir>/<local_dir.name>/<new_file>` (server-side copy + delete).
 
        If _validate=True:
          - The local file is also renamed to match the new S3 key.
          - A checksum is validated against the new S3 key.
          - Raises FileNotFoundError if no local copy exists to rename or validate against.
        """
        old_file = self._as_posix(old_file)
        new_file = self._as_posix(new_file)
        def _do():
            logger.info(f"[S3Client.rename] renaming '{old_file!r}' -> '{new_file!r}'")
            self._client.copy_object(
                Bucket=self._bucket,
                CopySource={"Bucket": self._bucket, "Key": self._key(old_file)},
                Key=self._key(new_file),
            )
            self._client.delete_object(Bucket=self._bucket, Key=self._key(old_file))
            
            if _validate:
                local_old = self._local(old_file)
                local_new = self._local(new_file)
                if local_old.exists():
                    local_old.rename(local_new)
                elif not local_new.exists():
                    raise FileNotFoundError(
                        f"Cannot validate rename: no local copy of '{old_file!r}' or '{new_file!r}' found."
                    )
                self._validate_checksum(new_file)
 
        return self._call(f"rename({old_file} → {new_file})", _do, _async=_async)
        
        
    def validate(
        self, 
        file: str | Path, 
        _async: bool = False
    ) -> bool | Future[bool]:
        """ Validate the local file's checksum against its S3 ETag. """
        file = self._as_posix(file)
        return self._call(f"validate({file})", lambda: self._validate_checksum(file), _async=_async)
        
        
    def sync(
        self, 
        file: str | Path, 
        _async: bool = False
    ) -> bool | Future[bool]:
        """
        Ensure the local directory and S3 have matching copies of filename,
        verified by checksum.
          - Only local exists: uploads to S3, then validates.
          - Only S3 exists:    download locally, then validates.
          - Both exist:        validates checksum only (no transfer).
          - Neither exists:    raises FileNotFoundError.
        """
        file = self._as_posix(file)
        def _do() -> None:
            local_exists = self._local(file).exists()
            try:
                self._client.head_object(Bucket=self._bucket, Key=self._key(file))
                s3_exists = True
            except ClientError:
                s3_exists = False
 
            if not local_exists and not s3_exists:
                raise FileNotFoundError(f"'{file!r}' not found locally or in S3.")
            elif not s3_exists:
                logger.info(f"[S3Client.sync] '{file!r}' not found on S3, uploading.")
                self._client.upload_file(
                    str(self._local(file)),
                    self._bucket,
                    self._key(file),
                    Config=self._TRANSFER_CONFIG,
                )
            elif not local_exists:
                logger.info(f"[S3Client.sync] '{file!r}' missing locally, fetching from S3.")
                self._local(file).parent.mkdir(parents=True, exist_ok=True)
                self._client.download_file(
                    self._bucket, self._key(file), str(self._local(file))
                )
 
            self._validate_checksum(file)
        return self._call(
            f"sync({file})", _do, _async=_async
        )
        
        
    def stream(self, file: str | Path) -> io.BytesIO:
        """ 
        Stream `S3://<bucket>/runs/<run-id>/<filename>` into an in-memory BytesIO buffer.
        Always raises on failure regardless of `strict` setting.
        """
        file = self._as_posix(file)
        try:
            logger.info(f"[S3Client.stream] streaming '{file!r}'")
            response = self._client.get_object(Bucket=self._bucket, Key=self._key(file))
            buffer = io.BytesIO(response["Body"].read())
            buffer.seek(0)
            return buffer
        except (ClientError, BotoCoreError) as e:
            raise RuntimeError(f"stream({file}) failed: {e}") from e
    
    
    def exists(self, file: str | Path) -> bool:
        file = self._as_posix(file)
        try:
            self._client.head_object(Bucket=self._bucket, Key=self._key(file))
            return True
        except ClientError:
            return False


    # --- utilities/lifecycle
    def _is_live(self) -> bool:
        try:
            boto3.client("sts", region_name=self._region).get_caller_identity()
            return True
        except (BotoCoreError, ClientError):
            return False
    
    def __enter__(self) -> "S3Client":
        return self

    def __exit__(self, *_):
        self._executor.shutdown(wait=True)
        
    def __del__(self) -> None:
        self._executor.shutdown(wait=False)
