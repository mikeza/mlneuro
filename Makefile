.PHONY: docs

docs:
	rm -rf docs/api docs/_build
	sphinx-apidoc -MeT -o docs/api mlneuro
	for f in docs/api/*.rst; do\
		perl -pi -e 's/(module|package)$$// if $$. == 1' $$f ;\
	done
	$(MAKE) -C docs html