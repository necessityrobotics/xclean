# The pre-registration of the final evaluation

`FINAL_BOARD_PREREG.md` is the pre-registration of our final evaluation: the plan, the tests and the rule for each
claim, written down before the final runs, with the amendments made during the evaluation. The
[results page](https://necessityrobotics.github.io/results.html) lists the claims and which of them passed, and the
amendments.

The file is published here exactly as it was fingerprinted on the results page: not one byte is changed. Its SHA-256
is:

```
ca7e7adebb492e6e51ac23fd3ab7002c51de3ff5309ed4429378621425355018
```

To check it, run one of these in the repository folder. Each prints the value above.

- Windows (Command Prompt or PowerShell): `certutil -hashfile docs\prereg\FINAL_BOARD_PREREG.md SHA256`
- Linux, Git Bash or WSL: `sha256sum docs/prereg/FINAL_BOARD_PREREG.md`
- macOS: `shasum -a 256 docs/prereg/FINAL_BOARD_PREREG.md`

Before you read it:

- It is an internal working document. It contains our internal working notes, internal names for parts of the
  engine and for tests, and the folder names of our machines, as they were written at the time. We publish it
  unedited so that anyone can check its fingerprint. Some short labels from it (for example P1.2, P1.3 and the p13_ run file names) also appear in the code.
- The file has mixed line endings (most lines end with CR LF, some with LF only), as it was written. The repository
  keeps them unchanged (the line `docs/prereg/** -text` in `.gitattributes`), so a clone gives the same SHA-256 on
  any system. Saving the file in an editor, or copying its text from a web page, can change the line endings and so
  the fingerprint: clone the repository or download the raw file.
- The fingerprint shows that this is the file that the results page fingerprinted. It does not by itself show when
  the file was written.
