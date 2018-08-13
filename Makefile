.PHONY: docs

docs:
	rm -rf docs/api docs/_build docs/_autosummary docs/generated
	# sphinx-apidoc -Me -o docs/api mlneuro
	# for f in docs/api/*.rst; do\
	# 	perl -pi -e 's/(module|package)$$// if $$. == 1' $$f ;\
	# done
	$(MAKE) -C docs html