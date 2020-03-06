.PHONY: all
all: proposal.pdf

proposal.pdf: proposal.tex
	latexmk -pdf $^

.PHONY: clean
clean:
	latexmk -C
