.PHONY: docs docs-noplot

docs:
	rm -rf docs/_build docs/_autosummary docs/generated
	$(MAKE) -C docs html

docs-noplot:
	rm -rf docs/_build docs/_autosummary docs/generated
	$(MAKE) -C docs html-noplot