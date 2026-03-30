import subprocess

with open('diff.txt', 'w', encoding='utf-8') as f:
    f.write(subprocess.check_output(['git', 'diff', 'HEAD~1', 'HEAD'], text=True))

with open('log.txt', 'w', encoding='utf-8') as f:
    f.write(subprocess.check_output(['git', 'log', '-n', '1', '--stat'], text=True))
