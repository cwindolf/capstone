figure:
	cd src && \
	if ! [ -a graphs.dat ] ; \
	then \
	     $(which python3) -m figures.make_some_graphs ; \
	fi;
	cd src && $(which python3) -m figures.figure2
	cd src && $(which python3) -m figures.figure3
	cd src && $(which python3) -m figures.figure4
	cd src && $(which python3) -m figures.figure5
	cd src && $(which python3) -m figures.figure6

clean:
	cd tex && rm -f *.aux *.bbl *.log *.pdf *.blg

pdf:
	cd tex && pdflatex capstone.tex && bibtex capstone && pdflatex capstone.tex && pdflatex capstone.tex

all:
	clean
	figure
	pdf
