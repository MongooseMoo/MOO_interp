#!/usr/bin/env python3
import sys

# Read the entire file
with open('moo_interp/builtin_functions.py', 'r') as f:
    lines = f.readlines()

# Find the insertion point (after file_tell, before # property functions)
insert_idx = None
for i, line in enumerate(lines):
    if '# property functions' in line and i > 1100:
        insert_idx = i
        break

if insert_idx is None:
    print("Could not find insertion point", file=sys.stderr)
    sys.exit(1)

# The new functions to insert
new_functions = '''
    def file_exists(self, path):
        """Check if a file exists.

        file_exists(path) => 1 if exists, 0 otherwise

        Requires wizard permissions.
        """
        # Type check
        if not isinstance(path, (str, MOOString)):
            raise MOOException(MOOError.E_TYPE, "file_exists() requires a string argument")

        # Convert to Python string
        path_str = str(path) if isinstance(path, MOOString) else path
        return 1 if os.path.exists(path_str) else 0

    def file_read(self, handle: int, bytes_to_read: int):
        """Read bytes from an open file handle.

        file_read(handle, bytes) => string data read
        """
        try:
            open_file = os.fdopen(handle)
            data = open_file.read(bytes_to_read)
            return MOOString(data)
        except (OSError, ValueError) as e:
            raise MOOException(MOOError.E_INVARG, f"file_read failed: {e}")

    def file_write(self, handle: int, data):
        """Write data to an open file handle.

        file_write(handle, data) => bytes written
        """
        try:
            open_file = os.fdopen(handle)
            data_str = str(data) if isinstance(data, MOOString) else str(data)
            bytes_written = open_file.write(data_str)
            return bytes_written if bytes_written is not None else len(data_str)
        except (OSError, ValueError) as e:
            raise MOOException(MOOError.E_INVARG, f"file_write failed: {e}")

    def file_eof(self, handle: int):
        """Check if file handle is at end of file.

        file_eof(handle) => 1 if at EOF, 0 otherwise
        """
        try:
            open_file = os.fdopen(handle)
            current_pos = open_file.tell()
            open_file.seek(0, os.SEEK_END)
            end_pos = open_file.tell()
            open_file.seek(current_pos)
            return 1 if current_pos >= end_pos else 0
        except (OSError, ValueError) as e:
            raise MOOException(MOOError.E_INVARG, f"file_eof failed: {e}")

    def file_stat(self, path):
        """Get file statistics.

        file_stat(path) => list [size, type, mode, owner, group, atime, mtime, ctime]
        """
        if not isinstance(path, (str, MOOString)):
            raise MOOException(MOOError.E_TYPE, "file_stat() requires a string argument")

        path_str = str(path) if isinstance(path, MOOString) else path

        try:
            st = os.stat(path_str)
            import stat as stat_module

            # Determine file type
            if stat_module.S_ISREG(st.st_mode):
                file_type = "reg"
            elif stat_module.S_ISDIR(st.st_mode):
                file_type = "dir"
            elif stat_module.S_ISCHR(st.st_mode):
                file_type = "chr"
            elif stat_module.S_ISBLK(st.st_mode):
                file_type = "block"
            elif stat_module.S_ISFIFO(st.st_mode):
                file_type = "fifo"
            elif stat_module.S_ISSOCK(st.st_mode):
                file_type = "socket"
            else:
                file_type = "unknown"

            mode_octal = oct(st.st_mode & 0o777)[2:]

            return MOOList([
                st.st_size,
                MOOString(file_type),
                MOOString(mode_octal),
                MOOString(""),
                MOOString(""),
                int(st.st_atime),
                int(st.st_mtime),
                int(st.st_ctime)
            ])
        except OSError as e:
            raise MOOException(MOOError.E_INVARG, f"file_stat failed: {e}")

    def file_type(self, path):
        """Get file type.

        file_type(path) => string ("reg", "dir", "chr", etc)
        """
        if not isinstance(path, (str, MOOString)):
            raise MOOException(MOOError.E_TYPE, "file_type() requires a string argument")

        path_str = str(path) if isinstance(path, MOOString) else path

        try:
            st = os.stat(path_str)
            import stat as stat_module

            if stat_module.S_ISREG(st.st_mode):
                return MOOString("reg")
            elif stat_module.S_ISDIR(st.st_mode):
                return MOOString("dir")
            elif stat_module.S_ISCHR(st.st_mode):
                return MOOString("chr")
            elif stat_module.S_ISBLK(st.st_mode):
                return MOOString("block")
            elif stat_module.S_ISFIFO(st.st_mode):
                return MOOString("fifo")
            elif stat_module.S_ISSOCK(st.st_mode):
                return MOOString("socket")
            else:
                return MOOString("unknown")
        except OSError as e:
            raise MOOException(MOOError.E_INVARG, f"file_type failed: {e}")

    def file_list(self, path, detailed: int = 0):
        """List directory contents.

        file_list(path [, detailed]) => list of filenames or detailed info
        """
        if not isinstance(path, (str, MOOString)):
            raise MOOException(MOOError.E_TYPE, "file_list() requires a string argument")

        path_str = str(path) if isinstance(path, MOOString) else path

        try:
            entries = os.listdir(path_str)
            result = []

            for entry in entries:
                if entry in ['.', '..']:
                    continue

                if detailed:
                    full_path = os.path.join(path_str, entry)
                    st = os.stat(full_path)
                    import stat as stat_module

                    if stat_module.S_ISREG(st.st_mode):
                        file_type = "reg"
                    elif stat_module.S_ISDIR(st.st_mode):
                        file_type = "dir"
                    elif stat_module.S_ISCHR(st.st_mode):
                        file_type = "chr"
                    elif stat_module.S_ISBLK(st.st_mode):
                        file_type = "block"
                    elif stat_module.S_ISFIFO(st.st_mode):
                        file_type = "fifo"
                    elif stat_module.S_ISSOCK(st.st_mode):
                        file_type = "socket"
                    else:
                        file_type = "unknown"

                    mode_octal = oct(st.st_mode & 0o777)[2:]

                    result.append(MOOList([
                        MOOString(entry),
                        MOOString(file_type),
                        MOOString(mode_octal),
                        st.st_size
                    ]))
                else:
                    result.append(MOOString(entry))

            return MOOList(result)
        except OSError as e:
            raise MOOException(MOOError.E_INVARG, f"file_list failed: {e}")

    def file_mkdir(self, path, mode: int = 0o777):
        """Create a directory.

        file_mkdir(path [, mode]) => 0 on success
        """
        if not isinstance(path, (str, MOOString)):
            raise MOOException(MOOError.E_TYPE, "file_mkdir() requires a string argument")

        path_str = str(path) if isinstance(path, MOOString) else path

        try:
            os.mkdir(path_str, mode)
            return 0
        except OSError as e:
            raise MOOException(MOOError.E_INVARG, f"file_mkdir failed: {e}")

    def file_rmdir(self, path):
        """Remove a directory.

        file_rmdir(path) => 0 on success
        """
        if not isinstance(path, (str, MOOString)):
            raise MOOException(MOOError.E_TYPE, "file_rmdir() requires a string argument")

        path_str = str(path) if isinstance(path, MOOString) else path

        try:
            os.rmdir(path_str)
            return 0
        except OSError as e:
            raise MOOException(MOOError.E_INVARG, f"file_rmdir failed: {e}")

    def file_remove(self, path):
        """Remove a file.

        file_remove(path) => 0 on success
        """
        if not isinstance(path, (str, MOOString)):
            raise MOOException(MOOError.E_TYPE, "file_remove() requires a string argument")

        path_str = str(path) if isinstance(path, MOOString) else path

        try:
            os.remove(path_str)
            return 0
        except OSError as e:
            raise MOOException(MOOError.E_INVARG, f"file_remove failed: {e}")

    def file_rename(self, old_path, new_path):
        """Rename a file or directory.

        file_rename(old, new) => 0 on success
        """
        if not isinstance(old_path, (str, MOOString)) or not isinstance(new_path, (str, MOOString)):
            raise MOOException(MOOError.E_TYPE, "file_rename() requires string arguments")

        old_str = str(old_path) if isinstance(old_path, MOOString) else old_path
        new_str = str(new_path) if isinstance(new_path, MOOString) else new_path

        try:
            os.rename(old_str, new_str)
            return 0
        except OSError as e:
            raise MOOException(MOOError.E_INVARG, f"file_rename failed: {e}")

    def file_copy(self, src, dst):
        """Copy a file.

        file_copy(src, dst) => 0 on success
        """
        import shutil

        if not isinstance(src, (str, MOOString)) or not isinstance(dst, (str, MOOString)):
            raise MOOException(MOOError.E_TYPE, "file_copy() requires string arguments")

        src_str = str(src) if isinstance(src, MOOString) else src
        dst_str = str(dst) if isinstance(dst, MOOString) else dst

        try:
            shutil.copy2(src_str, dst_str)
            return 0
        except OSError as e:
            raise MOOException(MOOError.E_INVARG, f"file_copy failed: {e}")

    def file_chmod(self, path, mode):
        """Change file permissions.

        file_chmod(path, mode) => 0 on success
        """
        if not isinstance(path, (str, MOOString)):
            raise MOOException(MOOError.E_TYPE, "file_chmod() requires path as string")

        path_str = str(path) if isinstance(path, MOOString) else path

        # Convert mode to integer
        if isinstance(mode, (str, MOOString)):
            mode_str = str(mode) if isinstance(mode, MOOString) else mode
            try:
                mode_int = int(mode_str, 8)
            except ValueError:
                raise MOOException(MOOError.E_INVARG, f"Invalid mode string: {mode_str}")
        elif isinstance(mode, int):
            mode_int = mode
        else:
            raise MOOException(MOOError.E_TYPE, "file_chmod() mode must be string or integer")

        try:
            os.chmod(path_str, mode_int)
            return 0
        except OSError as e:
            raise MOOException(MOOError.E_INVARG, f"file_chmod failed: {e}")

'''

# Insert the new functions
new_lines = lines[:insert_idx] + [new_functions] + lines[insert_idx:]

# Write back
with open('moo_interp/builtin_functions.py', 'w') as f:
    f.writelines(new_lines)

print(f"Inserted new FileIO builtins before line {insert_idx}")
