"""Workdir arguments refused before anything is read, and cleanup that cannot hide the error"""
from remolprobity import run


def test_a_file_as_workdir_is_refused(tmp_path, capsys):
    work = tmp_path / 'work'
    work.write_text('not a directory')
    assert run.main([str(tmp_path / 'missing.cif'), '-o', str(tmp_path / 'out.pkl'), '--workdir', str(work)]) == 1
    err = capsys.readouterr().err
    assert 'is not a directory' in err and 'reading' not in err
    assert work.read_text() == 'not a directory'


def test_cleanup_never_raises(tmp_path, monkeypatch):
    # _clean runs while an error is on its way out; its own failure must not replace it
    work = tmp_path / 'work'
    work.mkdir()
    (work / 'kept').write_text('')
    (work / 'added').write_text('')

    def remove(path):
        raise PermissionError('cannot remove %s' % path)

    monkeypatch.setattr(run.os, 'remove', remove)
    run._clean(str(work), False, {'kept'})
    assert sorted(p.name for p in work.iterdir()) == ['added', 'kept']
