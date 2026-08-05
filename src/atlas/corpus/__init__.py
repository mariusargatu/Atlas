"""Where documents and their gold set come from, and how they are cut into chunks.

Nothing here knows how retrieval works. The one way arrow this package sits behind is
`CollectionSource`: a source hands over documents, and `gold.resolve` turns the gold set
into the correct chunk set for whatever way of cutting is under test.
"""
