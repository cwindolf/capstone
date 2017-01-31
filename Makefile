figure:
	$(which python3) src/figure/make_all_figures.py

clean:
	cd fig && rm -f *.png
	cd tex && rm -f *.aux && rm *.bbl && rm *.log && rm *.pdf

pdf:
	clean
	figure
	cd tex && pdflatex capstone.tex && bibtex refs && pdflatex capstone.tex
