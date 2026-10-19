import pytest

from rinkspace.runtime_memory import (
    ProcessIdentity, ProcessMemory, ProcessTreeMemory, subtree_from_snapshot,
)


def process(pid, parent, token, rss):
    return ProcessMemory(ProcessIdentity(pid, token), parent, 'python.exe',
        rss, rss + 1, rss * 10, rss * 10, 'fixture')


def test_worker_snapshot_counts_only_its_actual_subtree():
    parent, worker, child, peer = (
        process(1, 0, 10, 1000), process(2, 1, 20, 100),
        process(3, 2, 30, 50), process(4, 1, 40, 200),
    )
    tree = ProcessTreeMemory((parent.identity,), (parent, worker, child, peer), (), 42.)
    result = subtree_from_snapshot(tree, worker.identity)
    assert result.processes == (worker, child)
    assert result.rss_bytes == 150
    assert result.private_bytes == 152
    assert result.sampled_at == 42.
    assert result.complete


def test_worker_snapshot_rejects_pid_reuse_or_incomplete_evidence():
    parent, worker = process(1, 0, 10, 1000), process(2, 1, 20, 100)
    tree = ProcessTreeMemory((parent.identity,), (parent, worker), (), 42.)
    with pytest.raises(MemoryError):
        subtree_from_snapshot(tree, ProcessIdentity(2, 999))
    with pytest.raises(MemoryError):
        subtree_from_snapshot(ProcessTreeMemory(tree.roots, tree.processes, ('unavailable',), 42.), worker.identity)
