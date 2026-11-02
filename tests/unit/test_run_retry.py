"""Which blocks run again, at what margin, and when a run must stop instead"""
import pytest

from remolprobity import run, schedule

DONE = {'block': 0, 'owned': []}


def asks(b, margin):
    return {'block': b, 'margin_needed': margin,
            'message': 'a reduce clique near the core (L 80, L 112) reaches within 5 Å of the margin; '
                       'use --margin %d or more' % margin}


def test_blocks_asking_for_a_larger_margin_run_again_at_it():
    assert run.retry_margins([DONE, asks(3, 23), asks(5, 21)], {}, 20.0) == {3: 23, 5: 21}


def test_nothing_to_retry():
    assert run.retry_margins([DONE], {}, 20.0) == {}


def test_up_to_twice_the_margin():
    assert run.retry_margins([asks(3, 40)], {}, 20.0) == {3: 40}
    assert run.retry_margins([asks(3, 30)], {3: 26}, 20.0) == {3: 30}


def test_beyond_twice_the_margin_stops_the_run_with_the_blocks_margin_error():
    with pytest.raises(schedule.BlockError) as e:
        run.retry_margins([DONE, asks(3, 41)], {}, 20.0)
    assert 'block 3 failed' in str(e.value)
    assert 'MarginError: a reduce clique near the core (L 80, L 112)' in str(e.value)
    assert 'use --margin 41 or more' in str(e.value)


def test_a_margin_not_above_the_blocks_own_stops_the_run():
    # the check would only repeat itself
    with pytest.raises(schedule.BlockError, match='block 3 failed'):
        run.retry_margins([asks(3, 23)], {3: 23}, 20.0)


def test_blocks_run_again_share_one_round_largest_first():
    # a round per margin left workers idle (8WQL: 8 blocks at 7 margins ran
    # mostly one at a time); each keeps its margin's block file and store
    again, size = {1: 26, 4: 26, 6: 40, 7: 34}, {1: 125697, 4: 121086, 6: 182535, 7: 146029}
    path = {b: '/w/blocks/block%05d.m%d.cif' % (b, m) for b, m in again.items()}
    tasks = run.retry_tasks(again, size, path, '/w', '/w/shared.pkl')
    assert [t['block'] for t in tasks] == [6, 7, 1, 4]
    assert [(t['margin'], t['store'], t['path']) for t in tasks] == [
        (m, '/w/results.m%d' % m, path[b]) for b, m in ((6, 40), (7, 34), (1, 26), (4, 26))]
    assert all(t['workdir'] == '/w' and t['shared'] == '/w/shared.pkl' for t in tasks)
