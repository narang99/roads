.PHONY: dvc-safe-clean-local-cache

dvc-safe-clean-local-cache:
	dvc gc --not-in-remote -a
