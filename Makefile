figure:
	cd src && $(which python3) -m figures.figure2
	cd src && $(which python3) -m figures.figure3
	cd src && $(which python3) -m figures.figure4
	cd src && $(which python3) -m figures.figure5
	cd src && $(which python3) -m figures.figure6

clean:
	# cd fig && rm -f *.png
	cd tex && rm -f *.aux *.bbl *.log *.pdf

pdf:
	cd tex && pdflatex capstone.tex && bibtex capstone && pdflatex capstone.tex && pdflatex capstone.tex

all:
	clean
	figure
	pdf
