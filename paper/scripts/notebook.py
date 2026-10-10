"""The paper as a Jupyter notebook, for working on the figures; the scripts stay the single source of truth.

    python scripts/notebook.py build [--execute]   tex + scripts -> paper.ipynb (text rendered as markdown, numbers filled in,
                                                   each figure's caption followed by its script as an editable cell)
    python scripts/notebook.py sync                edited figure cells -> scripts/fig_*.py (refuses to overwrite a
                                                   script that also changed on disk since the build; --force does)
    python scripts/notebook.py check               list the cells that differ from their scripts

Running a figure cell writes figures/<name>.pdf, as the script would, and shows the figure inline; `make paper` then
picks it up. The markdown cells are a rendering of tex/: edit the text there, and rebuild the notebook.
paper.ipynb is not in git (it contains DESI results); it is regenerated from the repository.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys

import nbformat

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TEX = os.path.join(ROOT, 'tex')
NB = os.path.join(ROOT, 'paper.ipynb')
SCRIPT_OF = {'diagonals_LRG1': 'fig_diagonals.py', 'diagonals_QSO': 'fig_diagonals.py'}   # else fig_<name>.py


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()[:16]


# ------------------------------------------------------------------ LaTeX -> markdown (the subset tex/ uses)
def braced(s, i):
    """content of the {...} group starting at s[i] == '{', and the index after it"""
    assert s[i] == '{', s[i:i + 30]
    depth, j = 0, i
    while True:
        if s[j] == '{' and s[j - 1] != '\\':
            depth += 1
        elif s[j] == '}' and s[j - 1] != '\\':
            depth -= 1
            if depth == 0:
                return s[i + 1:j], j + 1
        j += 1


def replace_cmd(s, cmd, fn, nargs=1):
    """replace \\cmd{a}{b}... by fn(a, b, ...) everywhere (nested groups allowed)"""
    out, i, pat = [], 0, '\\' + cmd + '{'
    while True:
        j = s.find(pat, i)
        if j < 0 or (s[j + len(pat) - 1:j + len(pat)] != '{'):
            out.append(s[i:])
            return ''.join(out)
        args, k = [], j + len(pat) - 1
        for _ in range(nargs):
            a, k = braced(s, k)
            args.append(a)
        out.append(s[i:j])
        out.append(fn(*args))
        i = k


def load_numbers():
    fn = os.path.join(TEX, 'numbers.tex')
    vals = {}
    if os.path.exists(fn):
        for m in re.finditer(r'\\csname val@(.+?)\\endcsname\{(.*?)\}\n', open(fn).read()):
            vals[m.group(1)] = m.group(2)
    return vals


def read_tex():
    main = open(os.path.join(TEX, 'main.tex')).read()
    title, _ = braced(main, main.index('\\title{') + 6)
    abstract, _ = braced(main, main.index('\\abstract{') + 9)
    body = main[main.index('\\begin{document}'):main.index('\\end{document}')]
    parts = []
    for m in re.finditer(r'\\input\{(sections/\w+)\}|\\appendix', body):
        parts.append('\\appendix' if m.group(0) == '\\appendix' else open(os.path.join(TEX, m.group(1) + '.tex')).read())
    return title, abstract, '\n\n'.join(parts)


def number_labels(text):
    """section / figure numbers by order of appearance, as LaTeX would assign them"""
    labels, sec, sub, fig, app = {}, 0, 0, 0, False
    for m in re.finditer(r'\\appendix|\\(section|subsection)\{|\\begin\{figure\}|\\label\{([^}]*)\}', text):
        t = m.group(0)
        if t == '\\appendix':
            app, sec = True, 0
        elif t.startswith('\\section'):
            sec, sub = sec + 1, 0
            cur = chr(64 + sec) if app else str(sec)
        elif t.startswith('\\subsection'):
            sub += 1
            cur = f'{chr(64 + sec) if app else sec}.{sub}'
        elif t.startswith('\\begin{figure'):
            fig += 1
            cur = str(fig)
        else:
            labels[m.group(2)] = cur
    return labels


def to_markdown(s, vals, labels):
    s = re.sub(r'(?<!\\)%.*', '', s)                                   # comments
    s = replace_cmd(s, 'val', lambda k: vals.get(k, f'**??{k}**'))
    for c in ('cref', 'Cref'):
        s = replace_cmd(s, c, lambda ks: ', '.join(
            ('Figure ' if k.startswith('fig:') else 'Section ' if k.startswith(('sec:', 'app:')) else '') +
            labels.get(k, '?') for k in ks.split(',')))
    s = replace_cmd(s, 'cite', lambda ks: '[' + ', '.join(ks.split(',')) + ']')
    s = replace_cmd(s, 'texorpdfstring', lambda a, b: a, nargs=2)
    s = replace_cmd(s, 'todo', lambda x: f'<span style="color:#e34948">[{x}]</span>')
    s = replace_cmd(s, 'emph', lambda x: f'*{x}*')
    s = replace_cmd(s, 'textbf', lambda x: f'**{x}**')
    s = replace_cmd(s, 'texttt', lambda x: f'`{x}`')
    s = replace_cmd(s, 'label', lambda x: '')
    s = replace_cmd(s, 'section', lambda x: f'\n## {x}\n')
    s = replace_cmd(s, 'subsection', lambda x: f'\n### {x}\n')
    s = replace_cmd(s, 'paragraph', lambda x: f'\n**{x}**')
    s = s.replace('\\appendix', '\n## Appendices\n')
    for a, b in (('\\thecov\\', '`thecov`'), ('\\thecov', '`thecov`'), ('\\hMpc', r'h\,\mathrm{Mpc}^{-1}'),
                 ('\\Mpch', r'h^{-1}\mathrm{Mpc}'), ('\\bk', r'\boldsymbol{k}'), ('\\bx', r'\boldsymbol{x}'),
                 ('\\br', r'\boldsymbol{r}'), ('\\noindent', ''), ('\\centering', ''), ('``', '“'), ("''", '”'),
                 ('---', '—'), ('--', '–'), ('\\%', '%'), ('~', ' ')):
        s = s.replace(a, b)
    s = re.sub(r'\\begin\{equation\}(.*?)\\end\{equation\}', lambda m: '\n$$' + m.group(1).strip() + '$$\n', s,
               flags=re.S)
    s = re.sub(r'\\begin\{(itemize|enumerate)\}|\\end\{(itemize|enumerate)\}', '\n', s)
    s = re.sub(r'^\s*\\item\s*', '\n- ', s, flags=re.M)
    s = re.sub(r'\n{3,}', '\n\n', s)
    return s.strip()


# ------------------------------------------------------------------ notebook
SETUP = '''# setup: run first. Figure cells below are the scripts in paper/scripts/ (edit here, then
#   python scripts/notebook.py sync    to write the edits back; `make paper` rebuilds the PDF).
import os, sys, importlib
sys.path.insert(0, os.path.abspath('scripts'))
os.chdir(os.path.abspath('scripts'))          # the scripts resolve products/ and figures/ relative to themselves
import common
importlib.reload(common)
from common import *'''


def cell_for_script(name, nb_scripts):
    fn = SCRIPT_OF.get(name, f'fig_{name}.py')
    path = os.path.join(HERE, fn)
    if not os.path.exists(path):
        return nbformat.v4.new_markdown_cell(f'*(no script `{fn}`)*')
    if fn in nb_scripts:
        return nbformat.v4.new_markdown_cell(f'*Made by the `{fn}` cell above.*')
    nb_scripts.add(fn)
    src = open(path).read()
    c = nbformat.v4.new_code_cell(src.rstrip('\n'))
    c.metadata['paper'] = dict(script=fn, sha=sha(src))
    c.metadata['tags'] = ['figure', fn]
    return c


def build(execute=False):
    vals = load_numbers()
    title, abstract, body = read_tex()
    labels = number_labels(body)
    cells = [nbformat.v4.new_markdown_cell(
        f'# {to_markdown(title, vals, labels)}\n\n'
        '*Notebook view of `paper/tex` (text, numbers from `tex/numbers.tex`) with the figure scripts as cells. '
        'Edit text in `tex/`; edit figures here and run `python scripts/notebook.py sync`.*\n\n'
        f'**Abstract.** {to_markdown(abstract, vals, labels)}'),
        nbformat.v4.new_code_cell(SETUP)]
    cells[-1].metadata['tags'] = ['setup']
    nb_scripts = set()
    pos = 0
    for m in re.finditer(r'\\begin\{figure\}(\[.*?\])?(.*?)\\end\{figure\}', body, flags=re.S):
        cells.append(nbformat.v4.new_markdown_cell(to_markdown(body[pos:m.start()], vals, labels)))
        env = m.group(2)
        names = re.findall(r'\\paperfig(?:\[[^\]]*\])?\{([^}]*)\}', env)
        cap, _ = braced(env, env.index('\\caption{') + 8)
        lab = re.search(r'\\label\{([^}]*)\}', env)
        num = labels.get(lab.group(1), '?') if lab else '?'
        cells.append(nbformat.v4.new_markdown_cell(f'> **Figure {num}.** {to_markdown(cap, vals, labels)}'))
        for n in names:
            cells.append(cell_for_script(n, nb_scripts))
        pos = m.end()
    cells.append(nbformat.v4.new_markdown_cell(to_markdown(body[pos:], vals, labels)))
    rest = sorted(set(f for f in os.listdir(HERE) if f.startswith('fig_')) - nb_scripts)
    if rest:
        cells.append(nbformat.v4.new_markdown_cell('## Figures not (yet) in the text'))
        for f in rest:
            cells.append(cell_for_script(f[4:-3], nb_scripts))
    tail = nbformat.v4.new_code_cell("%run make_numbers.py\n!cd .. && make paper    # rebuild tex/main.pdf")
    tail.metadata['tags'] = ['build']
    cells += [nbformat.v4.new_markdown_cell('## Rebuild numbers and the PDF'), tail]
    nb = nbformat.v4.new_notebook(cells=[c for c in cells if c.source.strip()])
    nb.metadata['kernelspec'] = dict(name='python3', display_name='Python 3', language='python')
    if execute:
        from nbclient import NotebookClient
        NotebookClient(nb, timeout=1800, resources={'metadata': {'path': ROOT}}, allow_errors=True,
                       skip_cells_with_tag='build').execute()
    nbformat.write(nb, NB)
    print(f'wrote {NB}: {len(nb.cells)} cells, {len(nb_scripts)} figure scripts')


def figure_cells():
    nb = nbformat.read(NB, as_version=4)
    for c in nb.cells:
        if c.cell_type == 'code' and 'paper' in c.metadata:
            yield c, os.path.join(HERE, c.metadata['paper']['script'])


def check():
    n = 0
    for c, path in figure_cells():
        disk = open(path).read()
        if c.source.rstrip('\n') != disk.rstrip('\n'):
            n += 1
            moved = sha(disk) != c.metadata['paper']['sha']
            print(f'{os.path.basename(path)}: notebook edited' + ('; script ALSO changed on disk since the build' if moved else ''))
    print(f'{n} cell(s) differ from their scripts')


def sync(force=False):
    nb = nbformat.read(NB, as_version=4)
    written = 0
    for c in nb.cells:
        if c.cell_type != 'code' or 'paper' not in c.metadata:
            continue
        path = os.path.join(HERE, c.metadata['paper']['script'])
        disk = open(path).read()
        new = c.source.rstrip('\n') + '\n'
        if new == disk:
            continue
        if sha(disk) != c.metadata['paper']['sha'] and not force:
            print(f'SKIPPED {os.path.basename(path)}: it changed on disk since the notebook was built (merge by hand, '
                  'or --force to overwrite)')
            continue
        open(path, 'w').write(new)
        c.metadata['paper']['sha'] = sha(new)
        written += 1
        print(f'wrote {os.path.basename(path)}')
    nbformat.write(nb, NB)
    print(f'{written} script(s) updated')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('action', choices=('build', 'sync', 'check'))
    ap.add_argument('--execute', action='store_true', help='build: run every figure cell (outputs inline)')
    ap.add_argument('--force', action='store_true', help='sync: overwrite scripts that changed on disk')
    a = ap.parse_args()
    {'build': lambda: build(a.execute), 'sync': lambda: sync(a.force), 'check': check}[a.action]()
