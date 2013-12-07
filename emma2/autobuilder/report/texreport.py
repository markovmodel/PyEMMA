'''
Created on Nov 20, 2013

@author: noe
'''

import os

class TexReport:
    """
    Utility class for generation of latex reports 
    """

    
    # basic text chunks
    documentheader = ("\\documentclass[12pt]{article}\n"
                      "\\usepackage{graphicx}\n"
                      "\\usepackage{float}"
                      "\n"
                      "\\title{MSM report}\n"
                      "\\date{}\n"
                      "\\begin{document}\n"
                      "\\maketitle\n"
                      "\n")
    
    documentfooter = "\\end{document}"
    
    plotcount = 0
    
    
    def __init__(self, 
                 name = "report",
                 directory = "./report"
                 ):
        
        print self.documentheader
        self.name = name;
        self.directory = directory;
        
        # create report directory if needed
        if not os.path.exists(directory):
            os.mkdir(directory)
        
        # start file
        self.tex = open(os.path.join(self.directory,self.name+".tex"), "w");
        self.text(self.documentheader)


    def section(self, header):
        self.tex.write("\\section{"+header+"}\n")

    def subsection(self, header):
        self.tex.write("\\subsection{"+header+"}\n")

    def paragraph(self, header):
        self.tex.write("\\paragraph{"+header+"}\n")

    def text(self, text):
        self.tex.write(text+"\n")

    def get_figure_name(self,ext):
        self.plotcount += 1
        return os.path.join(self.directory,str(self.plotcount)+"."+ext)

    def figure(self,filename,caption):
        self.tex.write("\n\\begin{figure}[h!]\n"
                      "\\begin{center}\n"
                      "\\includegraphics[width=0.8\textwidth]{"+filename+"}\n"
                      "\\caption{"+caption+"}\n"
                      "\\label{fig:"+filename+"}\n"
                      "\\end{center}\n"
                      "\\end{figure}\n"
                      "\n\n")


    def finish(self):
        self.text(self.documentfooter)
        self.tex.close();
        cwd = os.getcwd();
        os.chdir(self.directory)
        os.system("pdflatex "+self.name)
        os.system("pdflatex "+self.name)
        os.chdir(cwd)

