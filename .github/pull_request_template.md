**What this changes, and why**

**Did any published number move?**

If yes, say which one, what it went from and to, and the command that recorded it again. If a paid step (`--generate`, `--judge`, `--contrast`) is now stale because you could not afford to run it again, say that too. An admitted gap is fine. A stale number passed off as current is not.

- [ ] No published figure moved, or the ones that did were recorded again and named above.

**Checks**

CI fails on a fork because it needs repository secrets. Run it locally and say so here.

- [ ] `just lint` and `just types`
- [ ] `just test`
- [ ] If a prompt or the rubric changed, the version in the front matter, the field in the config, and the body hash in the judge test all moved together
- [ ] If tests were added, `tests/EXPECTED_MIN_TESTS` was bumped in this change
