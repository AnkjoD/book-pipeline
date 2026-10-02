"""Patch retriever.py: fix title/author extraction from 'books by X' queries."""
raw = open('src/serve/rag/retriever.py', 'rb').read()

# Locate the cand_title = re.sub block precisely
idx = raw.find(b'cand_title = re.sub')
assert idx > 0, "patch anchor not found"

# Find end of the if/filters block (ends before blank line)
end = raw.find(b'\n\n    # 2.', idx)
assert end > 0, "end anchor not found"

block = raw[idx:end]

# Build replacement (8-space indent to match file)
replacement = (
    b"cand_title = re.sub(\n"
    b"            r'^(?:books?|cu\xe1\xbb\x91n s\xc3\xa1ch|s\xc3\xa1ch)(?:\\s+|$)',\n"
    b"            '', cand_title, flags=re.IGNORECASE\n"
    b"        ).strip()\n"
    b"        # Always set author; only set title when it's a real title\n"
    b"        if cand_author:\n"
    b"            filters['author'] = cand_author\n"
    b"        if cand_title and cand_title.lower() not in ('book', 'books', 's\xc3\xa1ch'):\n"
    b"            filters['title'] = cand_title"
)

patched = raw[:idx] + replacement + raw[end:]
open('src/serve/rag/retriever.py', 'wb').write(patched)
print("patched OK, new block:")
print(patched[idx:idx+len(replacement)+10].decode('utf-8', errors='replace'))
