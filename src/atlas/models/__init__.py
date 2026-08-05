"""Everything that calls a model.

`embed` and `generate` hold the protocols, the wrappers and the bookkeeping that does not
depend on which provider answered; `providers` holds the clients and the seam that chooses
between them; `pricing` and `schemas` hold the two things that are data rather than
clients. The split is what lets a caller test the bookkeeping, substituting a fake client,
without an SDK present.
"""
