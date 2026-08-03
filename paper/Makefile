.PHONY: all clean check

all:
	latexmk -pdf -interaction=nonstopmode main.tex

# Pre-submission asserts.
#   1. no unfilled SLOT (repo URL / archive DOI)
#   2. no leftover placeholders
#   3. the manuscript is pure ASCII --- this also subsumes any check for
#      non-English internal vocabulary, which is why no word list is needed
#      here (a published word list would republish the words it excludes).
check:
	@# ⚠ measure what it means to measure: only occurrences that reach the
	@# rendered document count. A `%` comment describing the placeholder is not
	@# an unfilled slot --- an earlier version of this rule fired on its own
	@# documentation.
	@! grep -nE '^[^%]*SLOT-' main.tex \
		|| (echo "FAIL: SLOT remains unfilled (repo URL / archive DOI)" && exit 1)
	@! grep -nE 'TODO|FIXME|XXX' main.tex \
		|| (echo "FAIL: placeholder remains" && exit 1)
	@! LC_ALL=C grep -n '[^ -~	]' main.tex \
		|| (echo "FAIL: non-ASCII character in main.tex" && exit 1)
	@echo "check: OK"

clean:
	latexmk -C
