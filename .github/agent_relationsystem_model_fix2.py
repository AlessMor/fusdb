from pathlib import Path

replacements = {
    'tests/test_popcon_source_profiles.py': [
        ('_source_system(profile_size=31).compile()', '_source_system(profile_size=31)'),
        ('_source_system().compile()', '_source_system()'),
    ],
    'tests/test_popcon_source_profile_mappings.py': [
        ('_mapped_source_system(profile_size=31).compile()', '_mapped_source_system(profile_size=31)'),
        ('_mapped_source_system(fixed=True, profile_size=31).compile()', '_mapped_source_system(fixed=True, profile_size=31)'),
        ('_mapped_source_system().compile()', '_mapped_source_system()'),
    ],
}
for filename, pairs in replacements.items():
    path = Path(filename)
    text = path.read_text()
    for old, new in pairs:
        text = text.replace(old, new)
    path.write_text(text)
print('removed source-profile double compile calls')
