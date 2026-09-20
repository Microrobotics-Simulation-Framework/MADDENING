"""Two shipped cloud examples put MADDENING_API_TOKEN in a process argv.

src/maddening/examples/cloud/server/04_server_test.py:254 and
05_websocket_test.py:232 both do

    job.ssh_run_background(
        f"MADDENING_API_TOKEN={shlex.quote(API_TOKEN)} {PYTHON} /tmp/maddening_server.py")

and CloudJob.ssh_run_background (cloud/launcher.py:341) wraps that in

    nohup bash -c '<command>' > /tmp/bg_cmd.log 2>&1 &

before handing it to ssh as one argv element.  The credential is
therefore in the argv of the local ``ssh`` process, of the remote
``bash -c`` process, and in the remote shell's command record -- all of
which ``ps`` shows to every user on the box.  ``shlex.quote`` prevents
word splitting; it does nothing about visibility.

The library already has the out-of-band channel this should use
(MADDENING_API_TOKEN_FILE, written 0600), and ssh_run/ssh_run_background
expose no way to set an environment variable other than in the command
string.

This reproducer builds the exact command those examples produce and
shows it in a real /proc listing, using only a child process it starts
and reaps itself.
"""
import os, shlex, subprocess, time

API_TOKEN = "s3cret-bearer-token-not-a-real-one"
PYTHON = "python3"


def the_command_the_example_builds() -> str:
    inner = f"MADDENING_API_TOKEN={shlex.quote(API_TOKEN)} {PYTHON} /tmp/maddening_server.py"
    # cloud/launcher.py CloudJob.ssh_run_background
    return f"nohup bash -c {shlex.quote(inner)} > /tmp/bg_cmd.log 2>&1 &"


def main():
    cmd = the_command_the_example_builds()
    print("command handed to ssh as one argv element:")
    print(f"  {cmd}")
    print(f"  token present in it: {API_TOKEN in cmd}")

    # The same shape, locally.  NOTE: bash exec-optimises a single
    # command away, so the *final* process shows only "python ..." with the
    # token in /proc/<pid>/environ (owner-readable).  What is world-readable
    # is the shell that holds the whole command string -- on the operator's
    # machine that is the ``ssh`` process, for the life of the invocation.
    proc = subprocess.Popen(
        ["sh", "-c", f"nohup bash -c {shlex.quote('MADDENING_API_TOKEN=' + shlex.quote(API_TOKEN) + ' sleep 2')} >/dev/null 2>&1; true"]
    )
    time.sleep(0.3)
    try:
        with open(f"/proc/{proc.pid}/cmdline", "rb") as fh:
            argv = fh.read().replace(b"\0", b" ").decode(errors="replace")
        mode = oct(os.stat(f"/proc/{proc.pid}/cmdline").st_mode & 0o777)
        print()
        print(f"/proc/{proc.pid}/cmdline (mode {mode}, world-readable):")
        print(f"  {argv.strip()}")
        print(f"  token recoverable from the process listing: {API_TOKEN in argv}")
    finally:
        proc.wait(timeout=10)      # started here, reaped here; no kill by pattern


if __name__ == "__main__":
    main()
