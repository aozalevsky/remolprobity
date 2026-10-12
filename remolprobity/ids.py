"""Residue numbers and ids as remolprobity names residues: (chain, resid)"""


def residue_number(resseq: str) -> int:
    """
    A residue number given as text: decimal, or hybrid-36 as cctbx keeps
    numbers above 9999 in 4 characters (A000 is 10000; a000 follows ZZZZ)
    """
    s = resseq.strip()
    try:
        return int(s)
    except ValueError:
        pass
    value = int(s, 36)
    if s[0].isupper():
        return value - 10 * 36 ** 3 + 10000
    return value + 16 * 36 ** 3 + 10000


def residue_id(resseq: str, icode: str = '') -> str:
    """The residue number and insertion code: '42', '42A'"""
    return '%d%s' % (residue_number(resseq), icode.strip())


def residue_key(x) -> tuple:
    """(chain, resid) of a cctbx object with chain_id, resseq and icode"""
    return (x.chain_id.strip(), residue_id(x.resseq, x.icode))


def atom_tuple(x) -> tuple:
    """(chain, resid, resname, name, altloc) of a cctbx atom label or validation atom"""
    return (x.chain_id.strip(), residue_id(x.resseq, x.icode), x.resname.strip(),
            x.name.strip(), x.altloc.strip())
