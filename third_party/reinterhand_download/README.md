# Re:InterHand vendored download helpers

These files retain Meta's published capture and multipart file lists. The local
`download_utils.py` wrapper replaces shell-interpolated downloads and unsafe
archive extraction. For the maintained, selective pilot workflow, prefer:

```bash
python3 scripts/prepare_reinterhand_pilot.py --help
```

The vendored full-dataset scripts additionally require the external `wget` and
`tqdm` tools and may download very large archives. They are not imported by the
`hand_recon` package or exercised by the normal demo.
