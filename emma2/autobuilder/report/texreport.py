'''
Created on Nov 20, 2013

@author: noe
'''

import os
import socket
import datetime


class TexReport:
    """
    Utility class for generation of latex reports 
    """

    
    # basic text chunks
    documentheader = ("\\documentclass[12pt]{article}\n"
                      "\\usepackage{graphicx}\n"
                      "\\usepackage{float}\n"
                      "\\usepackage[top=2.5cm, bottom=2.5cm, left=2cm, right=2cm]{geometry}\n"
                      "\n"
                      "\\title{MSM report}\n"
                      "\\author{"+os.getlogin()+"@"+socket.gethostname()+"}\n"
                      "\\date{"+datetime.datetime.now().strftime("%Y-%m-%d %H:%M")+"}\n"
                      "\\begin{document}\n"
                      "\\maketitle\n"
                      "\n")
    
    documentfooter = ("\\clearpage\n"
                      "\n"
                      "\\bibliographystyle{plain}\n"
                      "\\bibliography{report}\n"
                      "\\end{document}")
    
    plotcount = 0
    
    
    def __init__(self, 
                 name = "report",
                 directory = "./report"
                 ):
        
        #print self.documentheader
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

    def cite(self, key):
        self.tex.write("\\cite{"+key+"}")

    def get_figure_name(self,ext):
        self.plotcount += 1
        return os.path.join(self.directory,str(self.plotcount)+"."+ext)

    def __table_row(self, row):
        ncol = len(row)
        self.tex.write(str(row[0]))
        for i in range(1,ncol):
            self.tex.write(" & "+str(row[i]))
        self.tex.write(" \\\\ \n")

    def table(self,tab,caption,align=None):
        ncol = len(tab[0])
        self.tex.write("\n\\begin{table}[h!]\n"
                      "\\begin{center}\n"
                      "\\begin{tabular}{\n")
        if (align != None):
            self.tex.write(align)
        else:
            for i in range(ncol):
                self.tex.write('c')
        self.tex.write("}")
        self.__table_row(tab[0])
        self.tex.write("\\hline\n")
        for i in range(1, len(tab)):
            self.__table_row(tab[i])
        self.tex.write("\\end{tabular}\n"
                       "\\caption{"+caption+"}\n"
                       #"\\label{tab:"+tail+"}\n"
                       "\\end{center}\n"
                       "\\end{table}\n"
                       "\n\n")

    def figure(self,filename,caption, width = 0.8):
        head,tail = os.path.split(filename)
        self.tex.write("\n\\begin{figure}[h!]\n"
                      "\\begin{center}\n"
                      "\\includegraphics[width="+str(width)+"\\textwidth]{"+tail+"}\n"
                      "\\caption{"+caption+"}\n"
                      "\\label{fig:"+tail+"}\n"
                      "\\end{center}\n"
                      "\\end{figure}\n"
                      "\n\n")

    def figure_mult(self,filenames,caption, width = 0.8):
        self.tex.write("\n\\begin{figure}[h!]\n"
                       "\\begin{center}\n")
        for filename in filenames:
            head,tail = os.path.split(filename)
            self.tex.write("\\includegraphics[width="+str(width)+"\\textwidth]{"+tail+"}\n")
        self.tex.write("\\caption{"+caption+"}\n"
                       "\\label{fig:"+tail+"}\n"
                       "\\end{center}\n"
                       "\\end{figure}\n"
                       "\n\n")


    def finish(self):
        self.text(self.documentfooter)
        self.tex.close();
        cwd = os.getcwd();
        os.chdir(self.directory)
        os.system("pdflatex "+self.name+" >& /dev/null")
        os.system("bibtex "+self.name+" >& /dev/null")
        os.system("pdflatex "+self.name+" >& /dev/null")
        os.system("pdflatex "+self.name+" >& /dev/null")
        os.chdir(cwd)

