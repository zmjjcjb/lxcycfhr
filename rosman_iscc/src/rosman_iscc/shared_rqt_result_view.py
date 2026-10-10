#!/usr/bin/env python3
"""Cross-node single-instance manager for the ISCC rqt_image_view window."""

import fcntl
import json
import os
import signal
import subprocess
import time


LOCK_PATH = '/tmp/rosman_iscc_rqt_result_view.lock'
STATE_PATH = '/tmp/rosman_iscc_rqt_result_view.json'
RQT_CMD = 'rqt_image_view'
RQT_STOP_WAIT_SEC = 1.0


def _log_info(message):
    try:
        import rospy

        rospy.loginfo(message)
    except Exception:
        pass


def _log_warn(message):
    try:
        import rospy

        rospy.logwarn(message)
    except Exception:
        pass


def _pid_exists(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except OSError as exc:
        return exc.errno != getattr(os, 'ESRCH', 3)
    except Exception:
        return False


def _owner_process_alive(owner_id):
    try:
        pid = int(str(owner_id).rsplit(':', 1)[1])
    except Exception:
        return True
    return _pid_exists(pid)


def _read_cmdline(pid):
    try:
        with open('/proc/%d/cmdline' % int(pid), 'rb') as handle:
            return handle.read().replace(b'\x00', b' ')
    except Exception:
        return b''


def _is_matching_rqt_process(pid):
    if not _pid_exists(pid):
        return False
    return b'rqt_image_view' in _read_cmdline(pid)


def _terminate_matching_rqt(pid):
    pid = int(pid)
    if not _is_matching_rqt_process(pid):
        return False
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        return False

    deadline = time.time() + RQT_STOP_WAIT_SEC
    while time.time() < deadline:
        if not _is_matching_rqt_process(pid):
            return True
        time.sleep(0.05)

    if _is_matching_rqt_process(pid):
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass

    deadline = time.time() + 0.5
    while time.time() < deadline:
        if not _is_matching_rqt_process(pid):
            return True
        time.sleep(0.05)
    return not _is_matching_rqt_process(pid)


def _normalize_state(data):
    state = {}

    topic = str(data.get('topic', '') or '').strip()
    if topic:
        state['topic'] = topic

    try:
        pid = int(data.get('pid', 0))
    except Exception:
        pid = 0
    if pid > 0:
        state['pid'] = pid

    owners = []
    seen = set()
    raw_owners = data.get('owners', [])
    if isinstance(raw_owners, (list, tuple)):
        for raw_owner in raw_owners:
            owner_id = str(raw_owner or '').strip()
            if not owner_id or owner_id in seen:
                continue
            if not _owner_process_alive(owner_id):
                continue
            owners.append(owner_id)
            seen.add(owner_id)
    if owners:
        state['owners'] = owners

    return state


def _load_state():
    if not os.path.exists(STATE_PATH):
        return {}
    try:
        with open(STATE_PATH, 'r') as handle:
            data = json.load(handle)
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    return _normalize_state(data)


def _write_state(state):
    state = _normalize_state(state)
    if not state.get('pid') and not state.get('owners'):
        try:
            os.remove(STATE_PATH)
        except FileNotFoundError:
            pass
        except OSError:
            pass
        return

    tmp_path = STATE_PATH + '.tmp'
    with open(tmp_path, 'w') as handle:
        json.dump(state, handle, sort_keys=True)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, STATE_PATH)


def _spawn_rqt(topic):
    with open(os.devnull, 'rb') as stdin_handle, open(os.devnull, 'ab') as output_handle:
        process = subprocess.Popen(
            [RQT_CMD, topic],
            stdin=stdin_handle,
            stdout=output_handle,
            stderr=output_handle,
            close_fds=True,
            start_new_session=True,
        )
    return int(process.pid)


def _lock_and_load_state():
    lock_fd = os.open(LOCK_PATH, os.O_CREAT | os.O_RDWR, 0o666)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        return lock_fd, _load_state()
    except Exception:
        os.close(lock_fd)
        raise


def _unlock(lock_fd):
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
    finally:
        os.close(lock_fd)


def acquire_shared_rqt_result_view(owner_id, topic):
    owner_id = str(owner_id or '').strip()
    topic = str(topic or '').strip()
    if not owner_id:
        raise ValueError('owner_id is required')
    if not topic:
        raise ValueError('topic is required')

    lock_fd, state = _lock_and_load_state()
    try:
        pid = int(state.get('pid', 0) or 0)
        current_topic = str(state.get('topic', '') or '').strip()
        owners = list(state.get('owners', []))

        if pid > 0 and not _is_matching_rqt_process(pid):
            _log_warn('shared rqt state had stale pid=%d, resetting' % pid)
            pid = 0
            state.pop('pid', None)

        if owner_id not in owners:
            owners.append(owner_id)
        state['owners'] = owners

        if pid > 0 and current_topic == topic:
            state['topic'] = topic
            state['pid'] = pid
            _write_state(state)
            return pid

        if pid > 0 and current_topic != topic:
            if current_topic:
                _log_info(
                    'shared rqt topic change detected: %s -> %s, restarting existing viewer'
                    % (current_topic, topic)
                )
            else:
                _log_warn('shared rqt state missing topic for pid=%d, restarting viewer' % pid)
            _terminate_matching_rqt(pid)
            state.pop('pid', None)

        new_pid = _spawn_rqt(topic)
        state['topic'] = topic
        state['pid'] = new_pid
        _write_state(state)
        _log_info('shared rqt_image_view active on %s (pid=%d)' % (topic, new_pid))
        return new_pid
    finally:
        _unlock(lock_fd)


def release_shared_rqt_result_view(owner_id):
    owner_id = str(owner_id or '').strip()
    if not owner_id:
        return False

    lock_fd, state = _lock_and_load_state()
    try:
        owners = [item for item in state.get('owners', []) if item != owner_id]
        pid = int(state.get('pid', 0) or 0)

        if owners:
            state['owners'] = owners
            if pid > 0 and _is_matching_rqt_process(pid):
                state['pid'] = pid
            else:
                state.pop('pid', None)
            _write_state(state)
            return False

        if pid > 0:
            _terminate_matching_rqt(pid)
        _write_state({})
        return True
    finally:
        _unlock(lock_fd)
